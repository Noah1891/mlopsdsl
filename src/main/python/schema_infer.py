import sys
import json
import pandas as pd

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
    df = pd.read_csv(path)
    columns = {}
    for col in df.columns:
        columns[col] = {
            "type": infer_type(df[col])
        }
    print(json.dumps({"columns": columns}))

if __name__ == "__main__":
    main()