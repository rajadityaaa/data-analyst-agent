"""
agent.py - LLM Agent Layer for CSV Data Analyst Agent.

This module handles LLM interactions using Google's Gemini API (google-generativeai).
It implements:
1. select_tool: Interprets natural language questions and selects one of the 5 Pandas tools.
2. explain_result: Generates a natural-language explanation strictly based on verified Pandas outputs.
3. run_agent: Orchestrates schema creation, tool selection, safe dispatch, execution, and explanation.
"""

import os
import json
import re
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import tools

load_dotenv()

# Explicit, safe tool dispatch dictionary (NO getattr, NO eval, NO exec)
TOOL_MAP = {
    "dataset_overview": tools.dataset_overview,
    "column_statistics": tools.column_statistics,
    "group_and_aggregate": tools.group_and_aggregate,
    "top_values": tools.top_values,
    "filter_data": tools.filter_data
}

MODEL_CANDIDATES = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"]


def _init_api():
    """Validates and configures the Gemini API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is missing or invalid. Please configure your .env file.")
    genai.configure(api_key=api_key)


def _extract_json(text: str) -> dict:
    """Extracts and parses JSON from model response text, stripping markdown blocks if needed."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def select_tool(question: str, schema: dict) -> dict:
    """Uses LLM to select an analysis tool and construct arguments matching the dataset schema."""
    try:
        _init_api()
    except Exception as e:
        return {"tool": None, "reason": str(e)}

    system_prompt = f"""You are a precise data analysis tool-selection controller.
You are given a user question and a dataset schema. Your job is to select EXACTLY ONE tool from the list below and provide arguments matching the schema.

AVAILABLE TOOLS:
1. "dataset_overview"
   - Arguments: {{}} (no arguments)
   - Description: Use when the question asks about overall dataset size, row count, column names, schema, or preview.

2. "column_statistics"
   - Arguments: {{"column": "<column_name>"}}
   - Description: Use when asking for summary statistics, average, median, min, max, standard deviation, or value counts of a single column.

3. "group_and_aggregate"
   - Arguments: {{"group_by": "<categorical_column>", "metric": "<numeric_column>", "operation": "<sum|mean|count|min|max>"}}
   - Description: Use when asking to compare, sum, average, min/max, or group a numeric metric across categories/regions/types/products.
   - Note: If operation is "count", metric can be any column. Otherwise, metric MUST be a numeric column.

4. "top_values"
   - Arguments: {{"column": "<column_name>", "n": <int>, "sort_by": "<numeric_column_or_null>", "ascending": <bool>}}
   - Description: Use when asking for top N rows/products/items by a column or metric.

5. "filter_data"
   - Arguments: {{"column": "<column_name>", "operator": "<==|!=|>|<|>=|<=|contains>", "value": <literal_value>}}
   - Description: Use when asking to filter rows where a column matches a condition or contains a keyword.

DATASET SCHEMA:
Columns and Dtypes:
{json.dumps(schema, indent=2)}

CRITICAL INSTRUCTIONS:
- You must ONLY select column names that exist in the DATASET SCHEMA above (exact case match).
- If the question references a column that does NOT exist in the schema, or cannot be answered by any available tool, return:
  {{"tool": null, "reason": "Reason why no tool can answer this question"}}
- NEVER compute numbers yourself. NEVER output Python code.
- Return ONLY valid JSON in this format:
{{"tool": "<tool_name_or_null>", "arguments": {{...}}, "reason": "<optional_reason_if_null>"}}
"""

    prompt = f"User Question: \"{question}\"\n\nRespond with ONLY the JSON tool selection object."

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"{system_prompt}\n\n{prompt}",
                generation_config={"temperature": 0.0}
            )
            return _extract_json(response.text)
        except Exception as e:
            last_error = e
            if "429" in str(e):
                time.sleep(2)
            continue

    return {"tool": None, "reason": f"Tool selection failed: {str(last_error)}"}


def explain_result(question: str, tool_result: dict) -> str:
    """Converts Pandas verified computation result into a structured natural-language response."""
    try:
        _init_api()
    except Exception as e:
        return f"Computation succeeded, but explanation failed: {str(e)}"

    system_prompt = """You are a professional data analyst explaining verified calculation results.
You are given a user question and verified numeric computation results produced by Pandas.

INSTRUCTIONS:
1. Explain the answer using ONLY the numbers provided in the verified computation result.
2. NEVER invent, hallucinate, or alter any numbers.
3. Structure your response into 3 clear sections:
   - **Answer**: A bold, direct 1-2 sentence response to the question.
   - **Evidence**: Key numeric facts / table summary supporting the answer.
   - **Insight**: (Optional) One short sentence contextualizing the finding.
4. Keep the response concise, professional, and clear.
"""

    prompt = f"""User Question: "{question}"

Verified Computation Result (Pandas Output):
{json.dumps(tool_result, indent=2)}

Generate the natural-language explanation adhering to the formatting instructions.
"""

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"{system_prompt}\n\n{prompt}",
                generation_config={"temperature": 0.2}
            )
            return response.text
        except Exception as e:
            last_error = e
            if "429" in str(e):
                time.sleep(2)
            continue

    return f"Failed to generate explanation: {str(last_error)}"


def run_agent(question: str, df: pd.DataFrame) -> dict:
    """Main agent function: schema creation -> tool selection -> safe dispatch -> explanation."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"status": "error", "message": "No valid dataset loaded."}

    if not question or not question.strip():
        return {"status": "error", "message": "Please enter a valid question."}

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Step 1: Tool Selection
    selection = select_tool(question.strip(), schema)

    tool_name = selection.get("tool")
    arguments = selection.get("arguments", {})
    reason = selection.get("reason", "No suitable tool found for this question.")

    if not tool_name:
        return {
            "status": "error",
            "message": f"I couldn't answer that from this dataset: {reason}",
            "tool_selected": None,
            "evidence": None
        }

    if tool_name not in TOOL_MAP:
        return {
            "status": "error",
            "message": f"The agent selected an unknown tool: '{tool_name}'",
            "tool_selected": tool_name,
            "evidence": None
        }

    # Step 2: Safe Dispatch via hardcoded dictionary
    tool_fn = TOOL_MAP[tool_name]

    try:
        if arguments:
            tool_result = tool_fn(df, **arguments)
        else:
            tool_result = tool_fn(df)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}",
            "tool_selected": tool_name,
            "evidence": None
        }

    # Check if tool returned error
    if isinstance(tool_result, dict) and "error" in tool_result:
        return {
            "status": "error",
            "message": tool_result["error"],
            "tool_selected": tool_name,
            "evidence": tool_result
        }

    # Step 3: Result Explanation
    explanation = explain_result(question, tool_result)

    return {
        "status": "success",
        "question": question,
        "tool_selected": tool_name,
        "arguments": arguments,
        "evidence": tool_result,
        "explanation": explanation
    }
