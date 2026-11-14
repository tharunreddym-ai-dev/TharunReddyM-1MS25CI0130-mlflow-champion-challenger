# 1MS25CI130 – Champion–Challenger Model Comparison with Hugging Face and MLflow
**Student:** Tharun Reddy M · **USN:** 1MS25CI130 · **Section:** B · Skill Lab-1 (CISL-1)

## 1. Task and models
| Item | Value |
|---|---|
| Task | Binary sentiment classification (negative / positive) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`, Hugging Face Hub) |
| Subset | 200 sentences, balanced 100 positive / 100 negative, seed 42 → `data/eval_inputs.csv` |
| Model A – current Champion | `distilbert-base-uncased-finetuned-sst-2-english` (~67M) |
| Model B – Challenger | `textattack/bert-base-uncased-SST-2` (~110M, outputs LABEL_0/LABEL_1 → mapped to negative/positive) |
| Device | CPU, batch size 16, max_length 128 |

## 2. How to run
```bash
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
python evaluate.py            # runs A and B on the same 200 inputs, logs everything, makes the decision
python register_champion.py   # registers both versions, sets @champion / @challenger aliases
python predict_from_mlflow.py # loads models:/1MS25CI130-sentiment-model@champion and predicts data/new_data.csv
```
MLflow experiment: `1MS25CI130-hf-sentiment-champion-challenger` · Registered model: `1MS25CI130-sentiment-model`

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 on the shared 200-sentence set — **higher is better**.
- **Margin:** B must beat A by **at least 0.01** macro-F1 (1 point = about 2 sentences out of 200; anything smaller is treated as noise).
- **Guardrail:** B is rejected if its average latency is **more than 5×** A's — **lower latency is better**.
- **Tie-breaker:** if |Δ macro-F1| < 0.01, keep the **faster** model.

## 4. Results
| Metric | A: DistilBERT-SST2 | B: BERT-base-SST2 | B − A |
|---|---:|---:|---:|
| Accuracy ↑ | 0.8950 | 0.9250 | +0.0300 |
| Macro precision ↑ | 0.8982 | 0.9254 | +0.0272 |
| Macro recall ↑ | 0.8950 | 0.9250 | +0.0300 |
| **Macro-F1 ↑** | **0.8948** | **0.9250** | **+0.0302** |
| Avg latency (ms/sample) ↓ | 51.27 | 76.23 | 1.49× |
| Total inference (s) ↓ | 10.25 | 15.25 | +4.99 |

The two models disagree on 12 of the 200 sentences; 5 are commented in `outputs/disagreements.csv`.

## 5. Decision – Champion: Model B (`textattack/bert-base-uncased-SST-2`)
The Challenger improved macro-F1 from 0.8948 to 0.9250, a gain of +0.0302. That is three times the 0.01 margin fixed before the run, so the improvement condition is met. Model B is slower, 76.2 ms per sentence against 51.3 ms, but that is only 1.49× the Champion's latency, well inside the 5× guardrail. Following the rule, Model B replaces Model A as Champion; it is registered as version 2 with the alias `@champion`, and DistilBERT (version 1) keeps the alias `@challenger`.

Is the gain meaningful? On 200 sentences, +0.03 means about 6 more correct answers. It is consistent across every metric (precision, recall and accuracy all move by about 3 points), and in the 12 disagreements B was right most of the time, mainly on negation ("I don't think I laughed out loud once") and long sentences where the complaint comes at the end. Still, a 200-sentence sample is small, so the true gap could be smaller.

Trade-offs: B is about 1.6× bigger (110M vs 67M parameters) and ~50% slower on CPU. For a batch or low-traffic service that cost is acceptable for 3 points of accuracy; for a very high-traffic, latency-critical service DistilBERT would still be a reasonable choice.

## 6. Limitations and declaration
- 200 SST-2 movie-review sentences only; results may differ on other domains (tweets, news, product reviews).
- Both models were fine-tuned on SST-2 training data, so this measures in-domain accuracy.
- Latency was measured on a shared 2-core CPU and varies between runs; the ratio is more reliable than the absolute ms.
- `predict_from_mlflow.py` never calls Hugging Face directly — prediction goes through the MLflow pyfunc wrapper (`hf_wrapper.py`), and the version / HF model ID are read from the registry tags. No run IDs are hard-coded.
- New data (`data/new_data.csv`, 8 sentences) was written by me and is not in the evaluation set.
- AI assistant used: Hark (AI assistant) helped write and debug the scripts and README; I can explain every line.

## 7. Repository structure
```
data/        eval_inputs.csv, new_data.csv
outputs/     predictions_*.csv, metrics_summary*.json, model_comparison.csv, disagreements.csv,
             confusion_*.png, metric_comparison.png, new_data_predictions.csv, *_terminal.txt
screenshots/ MLflow runs, compare view, summary run, registry aliases, predict_from_mlflow.py terminal output
evaluate.py, hf_wrapper.py, register_champion.py, predict_from_mlflow.py, requirements.txt, .gitignore
```
