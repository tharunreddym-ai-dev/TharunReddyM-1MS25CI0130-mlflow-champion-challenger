"""Run Model A (Champion) and Model B (Challenger) on the same SST-2 inputs and log everything to MLflow.
Student: Tharun Reddy M | USN: 1MS25CI130"""
import json, os, time, platform, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, torch, transformers
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hf_wrapper import HFSentimentModel

USN = "1MS25CI130"
SEED, N_PER_CLASS, BATCH, MAX_LEN = 42, 100, 16, 128
EXPERIMENT = f"{USN}-hf-sentiment-champion-challenger"
# Champion rule (written in README BEFORE the final run)
PRIMARY, MARGIN, LATENCY_LIMIT = "macro_f1", 0.01, 5.0
MODELS = {
    "A": {"short": "distilbert-sst2", "id": "distilbert-base-uncased-finetuned-sst-2-english",
          "role": "champion", "label_map": {"NEGATIVE": "negative", "POSITIVE": "positive"}},
    "B": {"short": "bert-base-sst2", "id": "textattack/bert-base-uncased-SST-2",
          "role": "challenger", "label_map": {"LABEL_0": "negative", "LABEL_1": "positive"}},
}
os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT)
torch.manual_seed(SEED)

# ---- 1. shared evaluation set (SST-2 validation, balanced, seed 42) ----
p = hf_hub_download("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(p)
ev = pd.concat([df[df.label == k].sample(N_PER_CLASS, random_state=SEED) for k in (0, 1)]).sample(frac=1, random_state=SEED)
ev = pd.DataFrame({"sample_id": [f"eval_{i:03d}" for i in range(1, len(ev) + 1)], "text": ev.sentence.str.strip(),
                   "reference": ev.label.map({0: "negative", 1: "positive"})})
ev.to_csv("data/eval_inputs.csv", index=False)
print(f"Evaluation set: {len(ev)} SST-2 sentences ({N_PER_CLASS} positive / {N_PER_CLASS} negative) -> data/eval_inputs.csv")

tags = {"student_usn": USN, "dataset_name": "stanfordnlp/sst2 (validation)",
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__}
results = {}

with mlflow.start_run(run_name="compare_sentiment") as parent:
    mlflow.set_tags(tags)
    for key, m in MODELS.items():
        run_name = f"{key}_sentiment_{m['short']}"
        print(f"\n=== {run_name}  ({m['id']}) ===")
        tok = AutoTokenizer.from_pretrained(m["id"])
        mdl = AutoModelForSequenceClassification.from_pretrained(m["id"]).eval()
        preds, scores, t0 = [], [], time.perf_counter()
        for i in range(0, len(ev), BATCH):
            enc = tok(ev.text[i:i + BATCH].tolist(), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                pr = mdl(**enc).logits.softmax(-1)
            s, idx = pr.max(-1)
            preds += [m["label_map"][mdl.config.id2label[int(j)]] for j in idx]; scores += s.tolist()
        total = time.perf_counter() - t0
        y, yh = ev.reference.tolist(), preds
        prec, rec, f1, _ = precision_recall_fscore_support(y, yh, average="macro", zero_division=0)
        met = {"accuracy": accuracy_score(y, yh), "macro_precision": prec, "macro_recall": rec, "macro_f1": f1,
               "avg_latency_ms": total / len(ev) * 1000, "total_inference_s": total}
        results[key] = {**met, "hf_model_id": m["id"], "run_name": run_name}
        for k2, v in met.items(): print(f"  {k2:18s} {v:.4f}")

        out = ev.copy(); out["prediction"] = preds; out["score"] = np.round(scores, 4); out["correct"] = out.prediction == out.reference
        pfile = f"outputs/predictions_{m['short']}.csv"; out.to_csv(pfile, index=False)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, yh, labels=["negative", "positive"]), display_labels=["negative", "positive"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{key}: {m['short']}"); fig.tight_layout(); cm = f"outputs/confusion_{m['short']}.png"; fig.savefig(cm, dpi=120); plt.close(fig)

        with mlflow.start_run(run_name=run_name, nested=True) as r:
            mlflow.set_tags({**tags, "role": m["role"]})
            mlflow.log_params({"hf_model_id": m["id"], "task": "sentiment-classification", "role": m["role"],
                               "max_length": MAX_LEN, "batch_size": BATCH, "eval_samples": len(ev), "seed": SEED, "device": "cpu"})
            mlflow.log_metrics(met)
            json.dump({k2: round(v, 4) for k2, v in met.items()}, open(f"outputs/metrics_summary_{m['short']}.json", "w"), indent=2)
            for f in (pfile, cm, f"outputs/metrics_summary_{m['short']}.json", "data/eval_inputs.csv"): mlflow.log_artifact(f)
            mdir = f"hf_models/{m['short']}"; tok.save_pretrained(mdir); mdl.save_pretrained(mdir)
            info = mlflow.pyfunc.log_model(name="model", python_model=HFSentimentModel(), code_paths=["hf_wrapper.py"],
                                           artifacts={"hf_model_dir": mdir},
                                           model_config={"label_map": m["label_map"], "max_length": MAX_LEN, "hf_model_id": m["id"]},
                                           signature=mlflow.models.infer_signature(pd.DataFrame({"text": ["a great movie"]}), pd.DataFrame({"text": ["a great movie"], "prediction": ["positive"], "score": [0.99]})),
                                           pip_requirements=["mlflow", "torch", "transformers", "pandas"])
            results[key].update(run_id=r.info.run_id, model_uri=info.model_uri)
            print(f"  logged model: {info.model_uri}")
        del tok, mdl; gc.collect()

    # ---- 2. comparison + decision (rule fixed before running) ----
    A, B = results["A"], results["B"]
    delta = B[PRIMARY] - A[PRIMARY]; ratio = B["avg_latency_ms"] / A["avg_latency_ms"]
    if ratio > LATENCY_LIMIT: winner, why = "A", f"Challenger is {ratio:.2f}x slower (> {LATENCY_LIMIT}x guardrail)"
    elif delta >= MARGIN: winner, why = "B", f"Challenger beats Champion by {delta:+.4f} macro-F1 (>= {MARGIN})"
    elif abs(delta) < MARGIN: winner = "A" if A["avg_latency_ms"] <= B["avg_latency_ms"] else "B"; why = f"|delta| {abs(delta):.4f} < {MARGIN}, faster model kept"
    else: winner, why = "A", f"Challenger is worse by {delta:+.4f} macro-F1"
    mlflow.log_params({"primary_metric": PRIMARY, "margin": MARGIN, "latency_guardrail_x": LATENCY_LIMIT})
    mlflow.log_metrics({"delta_macro_f1": delta, "latency_ratio_B_over_A": ratio})
    mlflow.set_tags({"winner": f"{winner}_{MODELS[winner]['short']}", "decision_reason": why})

    comp = pd.DataFrame({k: {m2: round(results[k][m2], 4) for m2 in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "avg_latency_ms", "total_inference_s"]} for k in "AB"})
    comp.columns = [f"A_{MODELS['A']['short']}", f"B_{MODELS['B']['short']}"]; comp["B_minus_A"] = (comp.iloc[:, 1] - comp.iloc[:, 0]).round(4)
    comp.to_csv("outputs/model_comparison.csv")
    json.dump({"decision": {"winner": winner, "reason": why, "delta_macro_f1": round(delta, 4), "latency_ratio": round(ratio, 3)},
               **{k: {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in results[k].items()} for k in "AB"}},
              open("outputs/metrics_summary.json", "w"), indent=2)
    # bar chart on every quality metric + latency chart
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    q = comp.loc[["accuracy", "macro_precision", "macro_recall", "macro_f1"]].iloc[:, :2]; q.plot.bar(ax=ax[0], rot=0); ax[0].set_ylim(0.8, 1); ax[0].set_title("Quality metrics (higher is better)")
    comp.loc[["avg_latency_ms"]].iloc[:, :2].plot.bar(ax=ax[1], rot=0); ax[1].set_title("Avg latency ms/sample (lower is better)")
    fig.tight_layout(); fig.savefig("outputs/metric_comparison.png", dpi=120); plt.close(fig)
    pa = pd.read_csv(f"outputs/predictions_{MODELS['A']['short']}.csv"); pb = pd.read_csv(f"outputs/predictions_{MODELS['B']['short']}.csv")
    dis = pa[pa.prediction != pb.prediction][["sample_id", "text", "reference"]].assign(pred_A=pa.prediction, pred_B=pb.prediction)
    dis.head(5).to_csv("outputs/disagreements.csv", index=False)
    for f in ("outputs/model_comparison.csv", "outputs/metrics_summary.json", "outputs/metric_comparison.png", "outputs/disagreements.csv", "README.md"):
        if os.path.exists(f): mlflow.log_artifact(f)
    json.dump({k: {"run_id": results[k]["run_id"], "hf_model_id": results[k]["hf_model_id"]} for k in "AB"} | {"winner": winner}, open("outputs/model_runs.json", "w"), indent=2)

print("\n=== Champion-Challenger decision ===")
print(comp.to_string())
print(f"Winner: {winner} ({MODELS[winner]['id']}) -> {why}")
print(f"Disagreements between A and B: {len(dis)} of {len(ev)}")
