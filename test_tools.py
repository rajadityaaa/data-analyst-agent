"""
test_tools.py - Lightweight unit & safety tests for CSV Data Analyst Agent.
"""

import pandas as pd
import tools
import agent


def test_csv_loading_and_overview():
    """Test 1: sample/sales.csv loads successfully and dataset_overview returns correct counts."""
    df = pd.read_csv("sample/sales.csv", encoding="latin1")
    overview = tools.dataset_overview(df)

    assert "error" not in overview
    assert overview["row_count"] == 9994
    assert overview["column_count"] == 21
    assert "Sales" in overview["columns"]
    assert "Region" in overview["columns"]


def test_group_and_aggregate_mathematical_precision():
    """Test 2: Proves numerical results come from Pandas computation by matching exact expected values."""
    mock_df = pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Furniture", "Furniture", "Furniture"],
        "Sales": [150.50, 349.50, 100.00, 200.00, 50.00]
    })

    result = tools.group_and_aggregate(mock_df, "Category", "Sales", "sum")

    assert "error" not in result
    results_dict = {item["group"]: item["value"] for item in result["results"]}
    
    # Assert exact mathematical sums: Electronics = 500.0, Furniture = 350.0
    assert results_dict["Electronics"] == 500.0
    assert results_dict["Furniture"] == 350.0


def test_top_values_and_column_statistics():
    """Test 3: top_values and column_statistics on known DataFrame."""
    mock_df = pd.DataFrame({
        "Item": ["A", "B", "C", "D"],
        "Score": [10, 40, 20, 30]
    })

    # Test column_statistics
    stats = tools.column_statistics(mock_df, "Score")
    assert stats["type"] == "numeric"
    assert stats["mean"] == 25.0
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0

    # Test top_values (top 2 by Score descending)
    top2 = tools.top_values(mock_df, "Item", n=2, sort_by="Score", ascending=False)
    items = [r["Item"] for r in top2["results"]]
    assert items == ["B", "D"]


def test_nonexistent_column_error_handling():
    """Test 4: Calling tools with non-existent column returns structured error dict, not uncaught exception."""
    mock_df = pd.DataFrame({"A": [1, 2, 3]})

    res1 = tools.column_statistics(mock_df, "FakeColumn")
    assert "error" in res1
    assert "FakeColumn" in res1["error"]

    res2 = tools.group_and_aggregate(mock_df, "FakeGroup", "A", "sum")
    assert "error" in res2

    res3 = tools.top_values(mock_df, "FakeCol")
    assert "error" in res3


def test_invalid_operator_and_operation_error_handling():
    """Test 5: Calling tools with invalid operation or operator returns structured error dict."""
    mock_df = pd.DataFrame({"Group": ["X", "Y"], "Val": [10, 20]})

    res_op = tools.group_and_aggregate(mock_df, "Group", "Val", "multiply_all")
    assert "error" in res_op
    assert "Invalid operation" in res_op["error"]

    res_filter = tools.filter_data(mock_df, "Group", "INVALID_OP", "X")
    assert "error" in res_filter
    assert "Invalid operator" in res_filter["error"]


def test_malformed_llm_json_handling():
    """Test 6: Malformed LLM JSON string is handled gracefully without crashing."""
    malformed_json = "```json\n{ 'tool': 'invalid_json', arguments: \n```"
    
    caught = False
    try:
        agent._extract_json(malformed_json)
    except Exception:
        caught = True
    assert caught, "Malformed JSON should raise JSON parsing exception"

    # Test agent fallback function
    result = agent.select_tool("Invalid question", {"Col": "int64"})
    assert "tool" in result


if __name__ == "__main__":
    print("Running Prompt 6 Test Suite...")
    test_csv_loading_and_overview()
    print("[PASS] Test 1: CSV Loading and Dataset Overview")
    test_group_and_aggregate_mathematical_precision()
    print("[PASS] Test 2: Mathematical Precision (Pandas Verified)")
    test_top_values_and_column_statistics()
    print("[PASS] Test 3: Top Values & Column Statistics")
    test_nonexistent_column_error_handling()
    print("[PASS] Test 4: Non-existent Column Error Handling")
    test_invalid_operator_and_operation_error_handling()
    print("[PASS] Test 5: Invalid Operator/Operation Error Handling")
    test_malformed_llm_json_handling()
    print("[PASS] Test 6: Malformed LLM JSON Error Handling")
    print("\nALL 6 TESTS PASSED SUCCESSFULLY!")

