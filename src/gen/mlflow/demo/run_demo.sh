#!/bin/bash
set -e

cd "$(dirname "$0")"

mlflow server \
  --backend-store-uri sqlite:////tmp/mlflow.db \
  --default-artifact-root ./mlruns \
  --port 5000 &
 
until curl -s http://localhost:5000/health > /dev/null; do
  sleep 1
done
echo "MLflow server is running."
 
mlflow run . -e main
 
python - << 'PYEOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
versions = client.get_latest_versions("demo", stages=["None"])
if versions:
    client.transition_model_version_stage(
        name="demo",
        version=versions[0].version,
        stage="Production"
    )
    print(f"demo v{versions[0].version} → Production")
PYEOF
 
mlflow run . -e serve -P port=5001