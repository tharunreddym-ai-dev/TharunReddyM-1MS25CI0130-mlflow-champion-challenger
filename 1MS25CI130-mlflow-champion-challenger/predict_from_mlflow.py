"""Load the Champion through its MLflow alias and predict on new data (no Hugging Face code here).
Student: Tharun Reddy M | USN: 1MS25CI130"""
import os, sys, mlflow, pandas as pd
from mlflow.tracking import MlflowClient

REGISTERED = "1MS25CI130-sentiment-model"
ALIAS = sys.argv[1] if len(sys.argv) > 1 else "champion"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mv = MlflowClient().get_model_version_by_alias(REGISTERED, ALIAS)
uri = f"models:/{REGISTERED}@{ALIAS}"
print(f"Loading {uri}")
print(f"  -> version {mv.version} | hf_model_id = {mv.tags.get('hf_model_id')} | run = {mv.tags.get('run_name')}")
model = mlflow.pyfunc.load_model(uri)

new = pd.read_csv("data/new_data.csv")
out = model.predict(new[["text"]])
out.insert(0, "sample_id", new.sample_id)
out["model_version"] = mv.version; out["hf_model_id"] = mv.tags.get("hf_model_id")
os.makedirs("outputs", exist_ok=True)
out.to_csv("outputs/new_data_predictions.csv", index=False)
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 140)
print(out[["sample_id", "text", "prediction", "score"]].to_string(index=False))
print("\nSaved outputs/new_data_predictions.csv")
