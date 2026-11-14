"""Custom MLflow pyfunc wrapper for a Hugging Face sentiment model (as in Lab 9)."""
import mlflow.pyfunc
import pandas as pd


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        path = context.artifacts["hf_model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        cfg = context.model_config or {}
        # map raw labels (e.g. LABEL_0/LABEL_1) to negative/positive
        self.label_map = cfg.get("label_map", {})
        self.max_length = int(cfg.get("max_length", 128))

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        with self.torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)
        score, idx = probs.max(-1)
        raw = [self.model.config.id2label[int(i)] for i in idx]
        labels = [self.label_map.get(r, r).lower() for r in raw]
        return pd.DataFrame({"text": texts, "prediction": labels,
                             "score": [round(float(s), 4) for s in score]})
