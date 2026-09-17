import sys
import json
import pandas as pd
from pathlib import Path

def infer_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        clean_series = series.dropna()
        distinct = clean_series.unique()
        if len(distinct) == 2 and set(distinct).issubset({0, 1}):
            return "boolean"
        if pd.api.types.is_integer_dtype(series) or (clean_series % 1 == 0).all():
            return "integer"
        return "float"
    return "categorical"

def main():
    path = sys.argv[1]
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(json.dumps({"status": "ERROR", "message": f"File not found: {Path(path).resolve()}"}))
    columns = {}
    for col in df.columns:
        columns[col] = {
            "type": infer_type(df[col])
        }
    print(json.dumps({"status": "SUCCESS", "columns": columns}))

if __name__ == "__main__":
    main()