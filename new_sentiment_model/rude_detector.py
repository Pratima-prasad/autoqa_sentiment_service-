import os
import re
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rude_detector_api")


MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    r"C:\Users\27yas\Downloads\latest_sentiment_analysis_model",
)
MAX_LEN = 128
DEFAULT_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ESCALATION_KEYWORDS_EN = [
    "rbi", "reserve bank of india", "npci", "irda", "cibil", "rti",
    "government", "regulatory", "regulator", "central bank",
    "national consumer helpline", "consumer authority of india",
    "grievance officer", "banking ombudsman",
    "legal action", "legal notice", "legal case", "legal team",
    "legal complaint", "legal proceeding", "legal process", "legal suit",
    "legally", "consumer court", "consumer forum", "consumer complaint",
    "consumer case", "consumer helpline", "consumer protection",
    "court", "court case", "file a case", "file case", "file a complaint",
    "lawsuit", "advocate", "sue", "suit against", "class action",
    "escalate", "escalation", "supervisor", "senior", "senior manager",
    "senior executive", "senior authority", "higher authority",
    "higher management", "senior management", "manager", "superior",
    "ceo", "top management",
    "social media", "facebook", "instagram", "twitter", "whatsapp",
    "media", "press", "public", "public platform", "go public",
    "post online", "negative review", "bad review",
    "police", "police station", "police complaint", "police case",
    "fir", "first information report",
    "harassment", "mental harassment", "threat", "threaten",
    "fraud", "cheating", "scam", "blackmail",
    "grievance", "national consumer", "irdai",
]

ESCALATION_KEYWORDS_HI = [
    "आरबीआई", "रिज़र्व बैंक", "एनपीसीआई", "उपभोक्ता अदालत", "उपभोक्ता फोरम",
    "उपभोक्ता शिकायत", "कानूनी कार्रवाई", "कानूनी नोटिस", "कोर्ट", "अदालत",
    "मुकदमा", "थाना", "पुलिस", "पुलिस स्टेशन", "पुलिस शिकायत",
    "वरिष्ठ", "वरिष्ठ अधिकारी", "पर्यवेक्षक", "सुपरवाइज़र", "उच्च अधिकारी",
    "प्रबंधक", "सोशल मीडिया", "धमकी", "परेशान", "उत्पीड़न", "मानसिक उत्पीड़न",
    "धोखाधड़ी", "फ़रज़ी", "फ्रॉड", "शिकायत", "एस्केलेट",
    "मुझे परेशान",
    "consumer adalat", "police station", "supervisor se baat", "senior se baat",
    "court case", "case daal", "case karunga", "legal karunga",
    "ghar tak aa gaya", "ghar pe aa jaunga", "family ko pareshan",
    "dhamki", "pareshan",
]

RUDE_KEYWORDS_EN = [
    "idiot", "idiots", "stupid", "dumb", "moron", "moronic", "fool", "foolish",
    "useless", "worthless", "incompetent", "pathetic", "nonsense", "rubbish",
    "shut up", "shut it", "shut your mouth", "waste of time", "waste of money",
    "waste of my time", "get lost", "who cares", "i don't care", "not my problem",
    "you people", "you guys are useless", "do your job", "learn to do your job",
    "ridiculous", "disgusting", "terrible service", "horrible service",
    "worst service", "worst company", "scammers", "cheaters", "liars",
    "you're lying", "stop lying", "nonsense answer", "brainless", "clueless",
    "shameless", "disgrace", "annoying", "irritating", "sick of this",
    "fed up", "done with this", "enough of this", "rubbish service",
    "garbage service", "trash service", "loser", "losers", "jerk",
]

RUDE_KEYWORDS_HI = [
    "बेवकूफ", "पागल", "गधा", "बकवास", "बेकार", "नालायक", "कमीना", "कमीने",
    "मूर्ख", "बदतमीज़", "बदतमीजी", "गाली", "गालियां", "चुप कर", "चुप रहो",
    "दिमाग खराब", "दिमाग है या नहीं", "शर्म नहीं आती", "बेशर्म", "घटिया",
    "फालतू", "टाइम वेस्ट", "नाटक बंद करो", "ड्रामा बंद करो",
    "bewakoof", "bewkoof", "pagal", "gadha", "gadhe", "bakwaas", "bakwas",
    "bekaar", "bekar", "nalayak", "kamina", "kamine", "murkh", "badtameez",
    "badtameezi", "chup kar", "chup ho ja", "chup raho", "dimag kharab",
    "dimaag kharab", "dimaag hai ya nahi", "dimag hai ya nahi",
    "sharam nahi aati", "besharam", "ghatiya", "faltu", "faaltu",
    "time waste", "natak band karo", "drama band karo", "bakwas band karo",
    "nikamma", "nikammi", "nalaayak", "harami", "kutta", "kutte",
    "chutiya", "chutiye", "saala", "saale", "bhosdi", "bhosdike", "BC", "MC", "Maderchod",
]

NEGATION_WINDOW_WORDS = [
    "nahi", "nahin", "na", "mat", "not", "no", "kabhi nahi", "bilkul nahi",
    "नहीं", "मत",
]
NEGATION_WINDOW_TOKENS = 4


def _kw_to_pattern(kw: str) -> str:
    escaped = re.escape(kw)
    if re.fullmatch(r"[a-z ]+", kw):
        return rf"\b{escaped}\b"
    return escaped


def _build_pattern(keywords):
    all_kw = sorted(set(k.strip().lower() for k in keywords if k.strip()),
                     key=len, reverse=True)
    pattern = re.compile("|".join(_kw_to_pattern(k) for k in all_kw), flags=re.IGNORECASE)
    return all_kw, pattern


ALL_ESCALATION_KEYWORDS, ESCALATION_PATTERN = _build_pattern(
    ESCALATION_KEYWORDS_EN + ESCALATION_KEYWORDS_HI
)
ALL_RUDE_KEYWORDS, RUDE_PATTERN = _build_pattern(
    RUDE_KEYWORDS_EN + RUDE_KEYWORDS_HI
)


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    text_lower = text.lower()
    before = text_lower[:match_start].split()[-NEGATION_WINDOW_TOKENS:]
    after = text_lower[match_end:].split()[:NEGATION_WINDOW_TOKENS]
    window = before + after
    return any(neg in window for neg in NEGATION_WINDOW_WORDS)


def detect_escalation(text: str) -> int:
    if not text:
        return 0
    return 1 if ESCALATION_PATTERN.search(text) else 0


def get_escalation_keywords(text: str):
    if not text:
        return []
    return [kw for kw in ALL_ESCALATION_KEYWORDS
            if re.search(_kw_to_pattern(kw), text, flags=re.IGNORECASE)]


def detect_rude_keyword(text: str) -> bool:
    if not text:
        return False
    for m in RUDE_PATTERN.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return True
    return False


def get_rude_keywords(text: str):
    if not text:
        return []
    found = []
    for m in RUDE_PATTERN.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            found.append(m.group(0).lower())
    return sorted(set(found))


def detect_shouting(raw_text: str, min_letters: int = 6, ratio_threshold: float = 0.7) -> bool:
    if not raw_text:
        return False
    letters = [c for c in raw_text if c.isalpha()]
    if len(letters) < min_letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= ratio_threshold


def detect_excessive_punctuation(raw_text: str) -> bool:
    if not raw_text:
        return False
    return bool(re.search(r"[!?]{2,}", raw_text))


def detect_elongation(raw_text: str) -> bool:
    if not raw_text:
        return False
    return bool(re.search(r"([a-zA-Z\u0900-\u097F])\1{2,}", raw_text))


def detect_tone_signals(raw_text: str) -> dict:
    return {
        "shouting": detect_shouting(raw_text),
        "excessive_punctuation": detect_excessive_punctuation(raw_text),
        "elongation": detect_elongation(raw_text),
    }


def has_rude_tone(raw_text: str) -> bool:
    signals = detect_tone_signals(raw_text)
    return signals["shouting"] or signals["excessive_punctuation"]


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return dev / len(letters)


HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "mera", "meri", "mere", "tumhe",
    "tumhara", "tumhari", "aap", "aapka", "kar", "karo", "kijiye", "raha",
    "rahi", "rahe", "bhai", "yaar", "bahut", "bilkul", "abhi", "kaise",
    "kyun", "kyu", "itna", "itni", "sab", "koi", "kuch", "hoga", "hogi",
    "matlab", "theek", "accha", "achha", "mujhe", "humein", "unko", "isko",
    "usko", "wala", "wali", "chahiye", "diya", "diyi", "gaya", "gayi",
}
DEVANAGARI_BUCKET_THRESHOLD = 0.30
HINGLISH_BUCKET_THRESHOLD = 0.12


def hinglish_score(text: str) -> float:
    toks = [t.strip(".,!?'\"") for t in text.lower().split()]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in HINGLISH_MARKERS)
    return hits / len(toks)


def register_bucket(text: str) -> str:
    if not text:
        return "english_or_other"
    if devanagari_ratio(text) >= DEVANAGARI_BUCKET_THRESHOLD:
        return "hindi_devanagari"
    if hinglish_score(text) >= HINGLISH_BUCKET_THRESHOLD:
        return "hinglish_roman"
    return "english_or_other"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



class RudeModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.rude_head = nn.Linear(hidden, 2)
        self.escalation_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return {
            "rude_logits": self.rude_head(cls),
            "escalation_logits": self.escalation_head(cls),
        }


def _validate_model_dir(model_dir: Path):
    required = ["config.json", "model.safetensors", "heads.pt", "tokenizer_config.json"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"MODEL_DIR '{model_dir}' is missing required files: {missing}. "
            f"Check the path — right now it's set to: {model_dir}"
        )


def load_detector(model_dir_str: str):
    model_dir = Path(model_dir_str).resolve()
    log.info(f"Loading model from: {model_dir}")
    _validate_model_dir(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    backbone = AutoModel.from_pretrained(str(model_dir))
    model = RudeModel(backbone)

    heads_path = model_dir / "heads.pt"
    heads = torch.load(heads_path, map_location=DEVICE)
    model.rude_head.load_state_dict(heads["rude_head"])
    model.escalation_head.load_state_dict(heads["escalation_head"])
    model.to(DEVICE)
    model.eval()

    threshold_path = model_dir / "threshold.txt"
    threshold = DEFAULT_THRESHOLD
    if threshold_path.exists():
        try:
            with open(threshold_path, "r", encoding="utf-8-sig") as f:
                threshold = float(f.read().strip())
            log.info(f"Loaded threshold from threshold.txt: {threshold}")
        except Exception as e:
            log.warning(f"Could not parse threshold.txt ({e}); using default {DEFAULT_THRESHOLD}")
    else:
        log.warning(f"threshold.txt not found at {threshold_path}; using default {DEFAULT_THRESHOLD}")

    log.info(f"Model loaded on device: {DEVICE}")
    return model, tokenizer, threshold


app = FastAPI(title="Rude/Escalation Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"model": None, "tokenizer": None, "threshold": DEFAULT_THRESHOLD}


@app.on_event("startup")
def _startup():
    model, tokenizer, threshold = load_detector(MODEL_DIR)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["threshold"] = threshold


class PredictRequest(BaseModel):
    text: str
    aux_threshold: Optional[float] = 0.5
    use_keyword_hybrid: Optional[bool] = True


class BatchPredictRequest(BaseModel):
    texts: List[str]
    aux_threshold: Optional[float] = 0.5
    use_keyword_hybrid: Optional[bool] = True


class PredictResponse(BaseModel):
    prediction: str
    rude_prob: float
    rude_keyword_hit: bool
    rude_keywords_matched: List[str]
    tone_signals: dict
    escalation: int
    escalation_prob: float
    escalation_keywords_matched: List[str]
    register: str


def _predict_one(text: str, aux_threshold: float, use_keyword_hybrid: bool) -> dict:
    model, tokenizer, threshold = _state["model"], _state["tokenizer"], _state["threshold"]
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, truncation=True, padding=True, max_length=MAX_LEN,
                     return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
    rude_prob = torch.softmax(outputs["rude_logits"], dim=1)[0, 1].item()
    esc_prob = torch.softmax(outputs["escalation_logits"], dim=1)[0, 1].item()

    keyword_hit = use_keyword_hybrid and detect_rude_keyword(text)
    tone = detect_tone_signals(text)
    tone_hit = use_keyword_hybrid and has_rude_tone(text)
    prediction = "rude" if (rude_prob >= threshold or keyword_hit or tone_hit) else "neutral"

    return {
        "prediction": prediction,
        "rude_prob": round(rude_prob, 4),
        "rude_keyword_hit": keyword_hit,
        "rude_keywords_matched": get_rude_keywords(text) if keyword_hit else [],
        "tone_signals": tone,
        "escalation": int(esc_prob >= aux_threshold or detect_escalation(text) == 1),
        "escalation_prob": round(esc_prob, 4),
        "escalation_keywords_matched": get_escalation_keywords(text),
        "register": register_bucket(cleaned),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "loading",
        "device": str(DEVICE),
        "threshold": _state["threshold"],
        "model_dir": MODEL_DIR,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="'text' must be non-empty")
    return _predict_one(req.text, req.aux_threshold, req.use_keyword_hybrid)


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.texts:
        raise HTTPException(status_code=400, detail="'texts' must be a non-empty list")
    return [_predict_one(t, req.aux_threshold, req.use_keyword_hybrid) for t in req.texts]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rude_detector_api:app", host="0.0.0.0", port=8000, reload=False)