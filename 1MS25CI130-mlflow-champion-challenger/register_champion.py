"""Register Model A and Model B and set the champion / challenger aliases from the decision in evaluate.py.
Student: Tharun Reddy M | USN: 1MS25CI130"""
import os, mlflow
from mlflow.tracking import MlflowClient

USN = "1MS25CI130"
EXPERIMENT = f"{USN}-hf-sentiment-champion-challenger"
REGISTERED = f"{USN}-sentiment-model"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)

# find the latest comparison run and its decision (no hard-coded run IDs)
parent = client.search_runs([exp.experiment_id], "attributes.run_name = 'compare_sentiment'", order_by=["attributes.start_time DESC"], max_results=1)[0]
winner = parent.data.tags["winner"].split("_")[0]          # "A" or "B"
print(f"Comparison run: {parent.info.run_id}  winner tag = {parent.data.tags['winner']}")
print(f"Reason: {parent.data.tags['decision_reason']}")

versions = {}
for key in ("A", "B"):
    child = client.search_runs([exp.experiment_id], f"tags.mlflow.parentRunId = '{parent.info.run_id}' and attributes.run_name LIKE '{key}_%'", max_results=1)[0]
    mv = mlflow.register_model(f"runs:/{child.info.run_id}/model", REGISTERED)
    hf_id = child.data.params["hf_model_id"]
    client.set_model_version_tag(REGISTERED, mv.version, "hf_model_id", hf_id)
    client.set_model_version_tag(REGISTERED, mv.version, "run_name", child.info.run_name)
    client.set_model_version_tag(REGISTERED, mv.version, "macro_f1", f"{child.data.metrics['macro_f1']:.4f}")
    versions[key] = (mv.version, hf_id, child.info.run_name)
    print(f"Registered {child.info.run_name:32s} -> {REGISTERED} v{mv.version}")

loser = "B" if winner == "A" else "A"
client.set_registered_model_alias(REGISTERED, "champion", versions[winner][0])
client.set_registered_model_alias(REGISTERED, "challenger", versions[loser][0])
print(f"\n@champion   -> v{versions[winner][0]}  {versions[winner][1]}")
print(f"@challenger -> v{versions[loser][0]}  {versions[loser][1]}")
