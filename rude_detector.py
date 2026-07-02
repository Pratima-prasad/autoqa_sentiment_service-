import os
import io
import re
import json
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.environ.get("MODEL_DIR", ".")
MAX_LEN = int(os.environ.get("MAX_LEN", 128))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rude_detector_backend")


def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

SEVERE_ABUSE_TERMS = {
    "behanchod", "behenchod", "bahenchod", "bhenchod", "bhosdike", "bhosdi ke",
    "bhosda", "madarchod", "maderchod", "chutiya", "chutiye",
    "chutiyapa", "gandu", "gaandu", "randi", "raand", "harami", "haramzada",
    "haramzadi", "kamina", "kamini", "saala kutta", "kutte ki aulad",
    "भेनचोद", "बहनचोद", "भोसड़ी", "भोसडी", "मादरचोद", "चूतिया", "चुतिया",
    "गांडू", "रांड", "हरामी", "हरामज़ादा", "कमीना",
}


def contains_severe_abuse(cleaned_text: str) -> Optional[str]:
    for term in SEVERE_ABUSE_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, cleaned_text):
            return term
    return None


def _read_threshold(model_dir: str) -> float:
    env_th = os.environ.get("THRESHOLD")
    if env_th is not None:
        try:
            return float(env_th)
        except ValueError:
            logger.warning(f"Invalid THRESHOLD env var '{env_th}', ignoring.")
    th_path = os.path.join(model_dir, "threshold.txt")
    if os.path.exists(th_path):
        with open(th_path) as f:
            try:
                return float(f.read().strip())
            except ValueError:
                pass
    return 0.5


class ModelState:
    """Holds the currently loaded model/tokenizer; supports hot reload."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.threshold = 0.5
        self.model_dir = None
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self, model_dir: str = MODEL_DIR):
        try:
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            logger.info(f"Loading tokenizer + model from {model_dir} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(DEVICE)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.model_dir = model_dir
            self.threshold = _read_threshold(model_dir)
            self.loaded = True
            self.load_error = None
            logger.info(
                f"Model loaded. device={DEVICE}, threshold={self.threshold}, "
                f"params={sum(p.numel() for p in model.parameters()):,}"
            )
        except Exception as e:
            self.loaded = False
            self.load_error = str(e)
            logger.exception("Failed to load model")
            raise

    def _encode(self, texts: List[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 64) -> List[dict]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        results = []
        for i in range(0, len(texts), batch_size):
            raw_chunk = texts[i : i + batch_size]
            chunk = [clean_text(t) for t in raw_chunk]
            enc = {k: v.to(DEVICE) for k, v in self._encode(chunk).items()}
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for orig, cleaned, prob in zip(raw_chunk, chunk, probs):
                matched_term = contains_severe_abuse(cleaned)
                if matched_term is not None:
                    # Hard override: known severe abuse always flags as rude,
                    # regardless of what the model scored it.
                    final_prob = max(float(prob), 0.99)
                    prediction = "rude"
                else:
                    final_prob = float(prob)
                    prediction = "rude" if final_prob >= self.threshold else "neutral"
                results.append(
                    {
                        "text": orig,
                        "cleaned_text": cleaned,
                        "rude_prob": round(final_prob, 4),
                        "prediction": prediction,
                        "matched_lexicon_term": matched_term,
                    }
                )
        return results


state = ModelState()

app = FastAPI(title="Hinglish Rude Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    try:
        state.load(MODEL_DIR)
    except Exception:
        logger.error("Startup model load failed; server will run but /predict will 503.")


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: List[str]


class ReloadRequest(BaseModel):
    model_dir: Optional[str] = None
    threshold: Optional[float] = None


def _require_loaded():
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. error={state.load_error}",
        )


@app.get("/health")
def health():
    return {
        "status": "ok" if state.loaded else "model_not_loaded",
        "loaded": state.loaded,
        "model_dir": state.model_dir,
        "threshold": state.threshold,
        "device": str(DEVICE),
        "load_error": state.load_error,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    _require_loaded()
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    result = state.predict_batch([req.text])[0]
    return result


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    _require_loaded()
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    results = state.predict_batch(req.texts)
    n_rude = sum(1 for r in results if r["prediction"] == "rude")
    return {
        "count": len(results),
        "rude_count": n_rude,
        "neutral_count": len(results) - n_rude,
        "results": results,
    }


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), text_col: str = "transcript"):
    """
    Upload a CSV, get back a CSV file download with a `sentiment` column
    (rude / neutral) plus the raw probability, appended to the original rows.
    """
    _require_loaded()
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if text_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{text_col}' not found. Available columns: {list(df.columns)}",
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    results = state.predict_batch(texts)

    out_df = df.copy()
    out_df["rude_prob"] = [r["rude_prob"] for r in results]
    # Use a distinct name so we don't silently overwrite an existing
    # "sentiment" column in the input CSV (e.g. ground-truth labels).
    out_df["predicted_sentiment"] = [r["prediction"] for r in results]  # "rude" / "neutral"

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8-sig")

    out_name = os.path.splitext(file.filename)[0] + "_predicted.csv"

    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.post("/reload")
def reload_model(req: ReloadRequest):
    model_dir = req.model_dir or MODEL_DIR
    state.load(model_dir)
    if req.threshold is not None:
        state.threshold = req.threshold
    return {
        "status": "reloaded",
        "model_dir": state.model_dir,
        "threshold": state.threshold,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))