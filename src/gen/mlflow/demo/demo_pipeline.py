import time
import joblib
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import mlflow
import mlflow.pyfunc
from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")

df = pd.read_csv(Path(__file__).parent.parents[3] / "data" / "demo.csv")

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)

numeric_features = ["age", "income", "score"]
categorical_features = ["city", "category"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5)),
])


DRIFT_RULES = [
    {"feature": "age", "window": 500, "threshold": 0.05},
]


def check_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    threshold: float,
) -> dict:
    ks_stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    drift_detected = p_value < threshold

    return {
        f"drift_{feature_name}_ks_stat": ks_stat,
        f"drift_{feature_name}_p_value": p_value,
        f"drift_{feature_name}_detected": int(drift_detected),
    }


def predict_with_monitoring(
    pipeline,
    inputs: pd.DataFrame,
    reference_data: pd.DataFrame,
    drift_rules: list,
) -> pd.Series:

    with mlflow.start_run(run_name="monitoring", nested=False):

        start = time.perf_counter()
        predictions = pipeline.predict(inputs)
        latency_ms = (time.perf_counter() - start) * 1000
        mlflow.log_metric("latency_ms", latency_ms)

        for rule in drift_rules:
            feature = rule["feature"]
            window = rule["window"]
            threshold = rule["threshold"]

            if feature not in reference_data.columns or feature not in inputs.columns:
                print(f"Feature '{feature}' not found.")
                continue

            current = inputs[feature].tail(window)

            drift_metrics = check_feature_drift(
                reference=reference_data[feature],
                current=current,
                feature_name=feature,
                threshold=threshold,
            )
            mlflow.log_metrics(drift_metrics)

            drift_flag = drift_metrics[f"drift_{feature}_detected"]
            p_val = drift_metrics[f"drift_{feature}_p_value"]

            if drift_flag:
                print(f"WARNING: drift detected in '{feature}' "
                      f"(p={p_val:.4f} < {threshold})")
            else:
                print(f"No drift in '{feature}' "
                      f"(p={p_val:.4f} >= {threshold})")

    return pd.Series(predictions)


class MonitoredPipeline(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        self.reference_data = pd.read_csv(context.artifacts["reference_data"])
        self.drift_rules = DRIFT_RULES
        self.buffer = []
        self.request_count = 0
        self.check_interval = 50
        self.max_window = max(r["window"] for r in self.drift_rules)

    def predict(self, context, model_input):
        self.buffer.append(model_input)
        self.request_count += 1

        if len(self.buffer) > self.max_window:
            self.buffer = self.buffer[-self.max_window:]

        enough_samples = len(self.buffer) >= self.max_window
        at_check_point = self.request_count % self.check_interval == 0

        if enough_samples and at_check_point:
            window = pd.concat(self.buffer, ignore_index=True)
            predict_with_monitoring(
                pipeline=self.pipeline,
                inputs=window,
                reference_data=self.reference_data,
                drift_rules=self.drift_rules,
            )

        return pd.Series(self.pipeline.predict(model_input))


mlflow.set_experiment("demo")

with mlflow.start_run() as training_run:

    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")

    assert accuracy >= 0.5, f"Accuracy {accuracy:.3f} below threshold 0.8"
    assert precision >= 0.5, f"Precision {precision:.3f} below threshold 0.9"

    clf = full_pipeline.named_steps["classifier"]
    mlflow.log_params(clf.get_params())
    mlflow.log_metrics({
        "accuracy":accuracy,
        "precision":precision,
    })

    joblib.dump(full_pipeline, "pipeline.joblib")
    
    reference_data_path = str(Path(__file__).parent.parents[3] / "data" / "demo.csv")

    mlflow.pyfunc.log_model(
        artifact_path="pipeline",
        python_model=MonitoredPipeline(),
        artifacts={
            "pipeline":"pipeline.joblib",
            "reference_data":reference_data_path,
        },
    )

    training_run_id = training_run.info.run_id

mlflow.register_model(
    model_uri=f"runs:/{training_run_id}/pipeline",
    name="demo"
)
print(f"Modell 'demo' registriert (run_id: {training_run_id}).")