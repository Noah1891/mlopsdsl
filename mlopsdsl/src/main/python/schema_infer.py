import sys
import json
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
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

def test_db_connection(db_url):
    """Check, if connection to database is possible"""
    engine = None
    try:
        engine = create_engine(db_url, connect_args={"connect_timeout": 5})
        with engine.connect():
            return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if engine is not None:
            engine.dispose()


def safe_url(db_url):
    try:
        return make_url(db_url).render_as_string(hide_password=True)
    except Exception:
        return "<invalid URL>"   # keine Rohausgabe, sonst könnte ein Passwort sichtbar werden


def main():
    path = sys.argv[1]
    db_url = sys.argv[2] if len(sys.argv) == 3 else None

    result = {"status": "SUCCESS", "columns": {}}

    try:
        df = pd.read_csv(path)
        result["columns"] = {col: {"type": infer_type(df[col])} for col in df.columns}
    except FileNotFoundError:
        print(json.dumps({"status": "FILE_ERROR", "message": f"File not found: {Path(path).resolve()}"}))
        return

    if db_url:
        db_ok, error = test_db_connection(db_url)
        if not db_ok:
            result["status"] = "DB_ERROR"
            result["message"] = f"Couldn't connect to database {safe_url(db_url)}: {error}"

    print(json.dumps(result))

if __name__ == "__main__":
    main()