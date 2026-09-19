import time
import logging
import joblib
import pandas as pd
from collections import deque
from scipy.stats import ks_2samp, chisquare
from fastapi import FastAPI, BackgroundTasks
from pydantic import create_model
from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.engine import make_url
import os

logger = logging.getLogger("uvicorn.error")

artifact = joblib.load("LogReg_model.pkl")
model = artifact["model"]
transformers = artifact["transformers"]
transform_log = artifact["transform_log"]
transformed_columns = artifact["transformed_columns"]
selected_features = artifact["selected_features"]
baseline_df = artifact["baseline_df"]
methods = artifact["methods"]
windows = artifact["windows"]
thresholds = artifact["thresholds"]

TABLE_NAME = "requests"

url = make_url(artifact["db_url"])
override = os.environ.get("DB_HOST_OVERRIDE")
if override and url.host in ("localhost", "127.0.0.1"):
   url = url.set(host=override)

engine = create_engine(url, pool_pre_ping=True)
requests_table = Table(TABLE_NAME, MetaData(), autoload_with=engine)

def store_request(payload: dict) -> None:
   try:
       row = {f: payload[f] for f in selected_features}
       with engine.begin() as conn:
           conn.execute(insert(requests_table), row)
   except Exception:
       logger.exception("Saving request input in database failed")

InputSchema = create_model("InputSchema", **{f: (float | str, ...) for f in selected_features})

drift_windows = {feature: deque(maxlen=window) for feature, window in zip(baseline_df.columns, windows)}

app = FastAPI()


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
   for action, feature, method in transform_log:
       key = f"{'scaler' if action=='scale' else 'imputer' if action=='fillna' else 'encoder'}_{feature}"
       tr = transformers[key]
       if action == "encode" and hasattr(tr, "get_feature_names_out"):
           encoded = tr.transform(df[[feature]])
           new_cols = tr.get_feature_names_out([feature])
           df = pd.concat([df.drop(columns=[feature]), pd.DataFrame(encoded, columns=new_cols, index=df.index)], axis=1)
       else:
           df[[feature]] = tr.transform(df[[feature]])
   return df.reindex(columns=transformed_columns, fill_value=0)


def compute_drift_score(method, feature, window_values, baseline_values):
   if method == 'KS':
       statistic, _ = ks_2samp(baseline_values, window_values)
   else: # ChiSquare
       base_counts = pd.Series(baseline_values).value_counts()
       cur_counts = pd.Series(window_values).value_counts().reindex(base_counts.index, fill_value=0)
       base_freq = base_counts / base_counts.sum()
       cur_freq = cur_counts / max(cur_counts.sum(), 1)
       statistic, _ = chisquare(f_obs=cur_freq * len(window_values), f_exp=base_freq * len(window_values))
   return statistic


@app.post("/predict")
def predict(item: InputSchema, background_tasks: BackgroundTasks):
   start = time.perf_counter()
   raw = item.dict()
   warnings = []

   for feature, window, method, threshold in zip(baseline_df.columns, windows, methods, thresholds):
       drift_windows[feature].append(raw[feature])
       if len(drift_windows[feature]) == window:
           score = compute_drift_score(method, feature, list(drift_windows[feature]), baseline_df[feature])
           if score > threshold:
               warnings.append(f"Drift detected on {feature}: score={score:.4f} exceeds threshold={threshold}")

   df = pd.DataFrame([raw])
   df = apply_transforms(df)
   pred = model.predict(df)[0]

   elapsed_ms = (time.perf_counter() - start) * 1000
   if elapsed_ms > 500:
       warnings.append(f"Latency {elapsed_ms:.1f}ms exceeded {500}ms")
   background_tasks.add_task(store_request, raw)
   result = {"prediction": pred.item() if hasattr(pred, "item") else pred}
   if warnings:
       result["warnings"] = warnings
   return result
    