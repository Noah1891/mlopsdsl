import sys
import json
import pandas as pd

def infer_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        distinct = series.dropna().unique()
        if len(distinct) == 2 and set(distinct).issubset({0, 1}):
            return "boolean"
        return "numeric"
    return "categorical"

def main():
    path = sys.argv[1]
    df = pd.read_csv(path)
    columns = {}
    for col in df.columns:
        columns[col] = {
            "type": infer_type(df[col]),
            "rowCount": int(df[col].count()),
            "cardinality": int(df[col].nunique(dropna=True))
        }
    print(json.dumps({"columns": columns}))

if __name__ == "__main__":
    main()