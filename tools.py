"""
tools.py - Safe Pandas-based analysis tools for the CSV Data Analyst Agent.

This module contains a fixed, safe set of Python/Pandas functions for performing
real computations on user-uploaded DataFrames. The LLM agent selects tool names
and arguments, but all numerical calculations are executed strictly within these functions.
"""

import pandas as pd
import numpy as np


def _clean_val(val):
    """Converts Pandas/Numpy types into standard Python JSON-serializable objects."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return str(val)


def dataset_overview(df: pd.DataFrame) -> dict:
    """Returns row count, column count, column names, dtypes, missing-value counts, and a small preview."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Invalid input DataFrame."}

    preview_df = df.head(5).copy()
    preview = preview_df.to_dict(orient="records")
    clean_preview = [{k: _clean_val(v) for k, v in row.items()} for row in preview]

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(count) for col, count in df.isnull().sum().items()},
        "preview": clean_preview
    }


def column_statistics(df: pd.DataFrame, column: str) -> dict:
    """Returns descriptive statistics for a specified column."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Invalid input DataFrame."}
    if not column or column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {list(df.columns)}"}

    series = df[column]
    missing_count = int(series.isnull().sum())

    if pd.api.types.is_numeric_dtype(series):
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {
                "column": column,
                "type": "numeric",
                "count": 0,
                "missing_count": missing_count,
                "message": "Column contains no non-null numeric values."
            }
        return {
            "column": column,
            "type": "numeric",
            "count": int(clean_series.count()),
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "std": float(clean_series.std()) if len(clean_series) > 1 else 0.0,
            "missing_count": missing_count
        }
    else:
        value_counts = series.value_counts(dropna=True).head(10)
        top_values = [{"value": str(k), "count": int(v)} for k, v in value_counts.items()]
        return {
            "column": column,
            "type": "categorical",
            "unique_count": int(series.nunique(dropna=True)),
            "top_values": top_values,
            "missing_count": missing_count
        }


def group_and_aggregate(df: pd.DataFrame, group_by: str, metric: str, operation: str = "sum") -> dict:
    """Groups by a column and aggregates a metric using sum/mean/count/min/max."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Invalid input DataFrame."}

    if not group_by or group_by not in df.columns:
        return {"error": f"Group-by column '{group_by}' not found. Available columns: {list(df.columns)}"}

    if not metric or metric not in df.columns:
        return {"error": f"Metric column '{metric}' not found. Available columns: {list(df.columns)}"}

    allowed_ops = ["sum", "mean", "count", "min", "max"]
    operation = str(operation).lower()
    if operation not in allowed_ops:
        return {"error": f"Invalid operation '{operation}'. Allowed operations are: {allowed_ops}"}

    if operation != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        return {"error": f"Metric column '{metric}' must be numeric for operation '{operation}'."}

    try:
        grouped = df.groupby(group_by, dropna=False)[metric]
        if operation == "sum":
            res = grouped.sum()
        elif operation == "mean":
            res = grouped.mean()
        elif operation == "count":
            res = grouped.count()
        elif operation == "min":
            res = grouped.min()
        elif operation == "max":
            res = grouped.max()

        res_sorted = res.sort_values(ascending=False)

        results_list = []
        for grp, val in res_sorted.items():
            grp_str = str(grp) if not pd.isna(grp) else "N/A"
            val_clean = int(val) if isinstance(val, (int, np.integer)) or operation == "count" else round(float(val), 4)
            results_list.append({"group": grp_str, "value": val_clean})

        return {
            "group_by": group_by,
            "metric": metric,
            "operation": operation,
            "results": results_list
        }
    except Exception as e:
        return {"error": f"Aggregation failed: {str(e)}"}


def top_values(df: pd.DataFrame, column: str, n: int = 5, sort_by: str = None, ascending: bool = False) -> dict:
    """Returns top n rows for a column, optionally sorted by another metric column."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Invalid input DataFrame."}

    if not column or column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {list(df.columns)}"}

    try:
        n = int(n)
        if n <= 0:
            return {"error": "Parameter 'n' must be a positive integer."}
        n = min(n, 50)
    except Exception:
        return {"error": "Parameter 'n' must be a positive integer."}

    target_sort_col = sort_by if sort_by else column
    if target_sort_col not in df.columns:
        return {"error": f"Sort column '{target_sort_col}' not found. Available columns: {list(df.columns)}"}

    try:
        sorted_df = df.sort_values(by=target_sort_col, ascending=ascending).head(n)
        records = sorted_df.to_dict(orient="records")
        clean_records = [{k: _clean_val(v) for k, v in row.items()} for row in records]

        return {
            "target_column": column,
            "sorted_by": target_sort_col,
            "n": n,
            "ascending": ascending,
            "results": clean_records
        }
    except Exception as e:
        return {"error": f"Top values extraction failed: {str(e)}"}


def filter_data(df: pd.DataFrame, column: str, operator: str, value) -> dict:
    """Filters rows matching condition (==, !=, >, <, >=, <=, contains)."""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Invalid input DataFrame."}

    if not column or column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {list(df.columns)}"}

    allowed_operators = ["==", "!=", ">", "<", ">=", "<=", "contains"]
    operator = str(operator).strip()
    if operator not in allowed_operators:
        return {"error": f"Invalid operator '{operator}'. Allowed operators are: {allowed_operators}"}

    try:
        series = df[column]
        if operator == "contains":
            val_str = str(value)
            mask = series.astype(str).str.contains(val_str, case=False, na=False)
        else:
            if pd.api.types.is_numeric_dtype(series):
                try:
                    num_val = float(value)
                    if operator == "==":
                        mask = series == num_val
                    elif operator == "!=":
                        mask = series != num_val
                    elif operator == ">":
                        mask = series > num_val
                    elif operator == "<":
                        mask = series < num_val
                    elif operator == ">=":
                        mask = series >= num_val
                    elif operator == "<=":
                        mask = series <= num_val
                except ValueError:
                    return {"error": f"Value '{value}' cannot be converted to numeric for column '{column}'."}
            else:
                val_str = str(value)
                if operator == "==":
                    mask = series.astype(str) == val_str
                elif operator == "!=":
                    mask = series.astype(str) != val_str
                elif operator == ">":
                    mask = series.astype(str) > val_str
                elif operator == "<":
                    mask = series.astype(str) < val_str
                elif operator == ">=":
                    mask = series.astype(str) >= val_str
                elif operator == "<=":
                    mask = series.astype(str) <= val_str

        filtered_df = df[mask]
        matching_count = len(filtered_df)

        preview = filtered_df.head(10).to_dict(orient="records")
        clean_preview = [{k: _clean_val(v) for k, v in row.items()} for row in preview]

        return {
            "column": column,
            "operator": operator,
            "value": value,
            "matching_rows": matching_count,
            "total_rows": len(df),
            "preview": clean_preview
        }
    except Exception as e:
        return {"error": f"Filtering failed: {str(e)}"}
