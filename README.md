# CSV Data Analyst Agent — Agentic AI MVP

An LLM-powered data analyst agent that uses Python and Pandas tools to analyze CSV datasets through natural-language questions.

---

## 1. Project Title

**CSV Data Analyst Agent**

---

## 2. Description

The **CSV Data Analyst Agent** is a lightweight, agentic AI application that lets users upload any CSV dataset and ask questions in plain English. Instead of trusting an LLM to guess numerical figures or write unconstrained code, this application uses Google's Gemini LLM purely as a **reasoning controller**. The LLM interprets the user's question, selects from a fixed allow-list of safe Pandas analysis tools, and lets Python perform the actual computation. The LLM then translates the verified computation output into a structured, natural-language explanation with optional visualizations.

---

## 3. Why This Project Exists

Most non-technical users want quick insights from CSV files without writing code in Jupyter notebooks or SQL. Conversely, many naive AI data analyst tools paste raw data into LLM prompts and trust the model's math — a approach that is unreliable, error-prone, and insecure.

This project bridges that gap by combining natural-language interaction with **guaranteed-correct computation**. The LLM never invents numbers and never executes arbitrary Python code: numeric truth remains 100% in Pandas.

---

## 4. Features

- **CSV Dataset Upload & Validation**: Fast CSV parsing with automatic size/row limits (max 10MB, max 50,000 rows) and `latin1` encoding fallback.
- **Dataset Overview Panel**: Instant summary showing total rows, columns, data types, missing value counts, and a 5-row interactive preview.
- **Agentic Tool Selection**: LLM evaluates questions against dataset schemas and maps intent into structured JSON tool calls.
- **Verified Pandas Computation**: 5 safe analysis tools for overview, column statistics, grouping/aggregating, top values, and filtering.
- **Structured Explanations**: Converts verified numeric outputs into clear **Answer**, **Evidence**, and **Insight** sections.
- **Automated Visualization**: Rule-based Matplotlib bar and line charts rendered automatically for multi-category and top-N query results.
- **Comprehensive Error Handling**: Graceful error handling for missing API keys, invalid columns, malformed JSON, and off-topic questions.

---

## 5. Architecture Diagram

```text
               +-----------------------------+
               |           User              |
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               |        Streamlit UI         |
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               |    LLM Call #1: Select Tool  |
               | (Gemini JSON Tool Controller)|
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               |   Tool Dispatch (tools.py)  |
               | (Explicit Hardcoded Dict)   |
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               |   Pandas Computation Engine |
               | (Exact Mathematical Output) |
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               | LLM Call #2: Explain Result |
               |   (Natural Language Synthesis|
               +--------------+--------------+
                              |
                              v
               +--------------+--------------+
               |    Streamlit Results UI     |
               | (Answer/Evidence/Chart)     |
               +-----------------------------+
```

---

## 6. Agent Workflow Explanation

1. **User Question**: The user enters a question (e.g. *"Which region generated the highest revenue?"*).
2. **Schema Construction**: The app extracts column names and data types from the active Pandas DataFrame.
3. **Reasoning & Tool Selection**: Gemini analyzes the question and schema, selecting a tool (e.g. `group_and_aggregate`) and structured arguments (`{"group_by": "Region", "metric": "Sales", "operation": "sum"}`).
4. **Safe Computation**: Python dispatches arguments to `tools.py`, where Pandas computes the exact sum per region.
5. **Natural-Language Explanation**: Gemini receives the verified Pandas output dictionary and formats a concise response into **Answer**, **Evidence**, and **Insight**.
6. **Visualization**: A rule-based Matplotlib renderer checks the result shape and automatically generates a bar or line chart.

> **Core Principle**: The LLM decides *what* analysis to perform and *how to phrase the result*. Python and Pandas perform all calculations. The LLM never computes statistics directly and never generates executable code.

---

## 7. Tool-Calling Explanation

`tools.py` contains five fixed, safe Pandas analysis functions:

1. **`dataset_overview(df)`**: Returns total row/column counts, schema names, dtypes, missing value counts, and preview rows.
2. **`column_statistics(df, column)`**: Calculates count, mean, median, min, max, std, and missing counts for numeric columns, or top value counts for categorical columns.
3. **`group_and_aggregate(df, group_by, metric, operation)`**: Groups data by a column and aggregates a metric (`sum`, `mean`, `count`, `min`, `max`), returning sorted results.
4. **`top_values(df, column, n, sort_by, ascending)`**: Extracts the top $N$ rows sorted by a specified column.
5. **`filter_data(df, column, operator, value)`**: Filters rows using safe operators (`==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`) and returns match count + preview.

---

## 8. Technology Stack

- **`streamlit`**: Interactive web app framework for file upload, user interaction, and data display.
- **`pandas`**: High-performance data manipulation engine executing all real dataset calculations.
- **`matplotlib`**: Plotting library for rendering rule-based bar and line visualizations.
- **`google-generativeai`**: Google Gemini API SDK for LLM-based tool selection and result explanation.
- **`python-dotenv`**: Securely loads environment variables (`GEMINI_API_KEY`) from `.env`.

---

## 9. Project Structure

```text
csv-data-analyst-agent/
│
├── app.py              # Streamlit UI, file uploader, overview, and visualization rendering
├── agent.py            # LLM agent layer (Gemini tool selection + explanation generation)
├── tools.py            # Safe, validated Pandas computational analysis functions
├── test_tools.py       # Unit test suite verifying safety, numerical precision, and error handling
├── requirements.txt    # Pinned Python dependencies
├── .env.example        # Environment variable placeholder template
├── .gitignore          # Excludes secrets (.env), venv, bytecode, and OS files
├── README.md           # Project documentation and architecture guide
└── sample/
    ├── sales.csv        # Superstore retail dataset (~10,000 rows, 21 columns)
    └── retail_sales.csv # Retail transaction sample dataset
```

---

## 10. Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/csv-data-analyst-agent.git
   cd csv-data-analyst-agent
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 11. Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

---

## 12. Usage Instructions

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser at `http://localhost:8501`.
3. Upload a CSV file (or use `sample/sales.csv`).
4. Inspect the **Dataset Overview** panel.
5. Type a natural-language question in the input box and click **Analyze**.

---

## 13. Example Questions

1. *"Which region generated the highest total sales?"*
2. *"What are the top 5 products by revenue?"*
3. *"What is the average profit across all orders?"*
4. *"Which product category has the most orders?"*
5. *"Show me total sales grouped by Segment."*

---

## 14. Example Output

### Question: *"Which region generated the highest total sales?"*

**Tool Selected**: `group_and_aggregate` | **Arguments**: `{"group_by": "Region", "metric": "Sales", "operation": "sum"}`

#### **Answer**
**The West region generated the highest total sales, reaching $725,457.82.**

#### **Evidence**
- **West**: $725,457.82
- **East**: $678,781.24
- **Central**: $501,239.89
- **South**: $391,721.91

#### **Insight**
The West region outperformed the second-highest region (East) while generating nearly double the revenue of the lowest region (South).

---

## 15. Safety Considerations

- **No Arbitrary Code Execution**: The LLM cannot generate or execute Python code strings. All dynamic execution methods (`eval()`, `exec()`) are strictly forbidden.
- **Fixed Tool Allow-List**: Tool dispatch uses a hardcoded dictionary (`TOOL_MAP`). `getattr()` on user/LLM input is never used.
- **In-Memory Data Processing**: Uploaded CSV data lives strictly in Streamlit session state memory and is never written to disk or sent to third-party databases.
- **Secret Isolation**: `GEMINI_API_KEY` is loaded from `.env` via `python-dotenv` and excluded from source control via `.gitignore`.

---

## 16. Limitations

- **Single File Only**: Analyzes one CSV file per session; does not support multi-file joins or SQL databases.
- **Stateless Agent**: Operates single-turn (no multi-turn chat memory).
- **Dataset Size Limits**: Enforces a maximum limit of 50,000 rows and 10MB per CSV to guarantee fast response times.
- **Portfolio Scope**: Designed as an educational/portfolio MVP demonstrating tool-calling architecture, not a enterprise-scale analytics backend.

---

## 17. Future Improvements

- Add conversation memory for follow-up contextual questions.
- Support multi-file CSV uploads and automatic table joins.
- Add advanced statistical tools (correlation matrix, outlier detection, time-series forecasting).
- Support export options (download analysis report as PDF or Markdown).

---

## 18. Screenshots Placeholder

*(Place application UI screenshots here showing dataset overview, agent analysis, and Matplotlib visualizations)*

---

## 19. License

This project is open-source and available under the [MIT License](LICENSE).
