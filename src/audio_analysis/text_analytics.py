"""Motor NLP Clínico Open Source para análise de sentimento e termos críticos (PT-BR e EN)."""
from __future__ import annotations

from typing import Any, Dict, List

CRITICAL_CLINICAL_TERMS = {
    # Português
    "falta de ar": 0.90,
    "dispneia": 0.90,
    "dor no peito": 0.95,
    "dor torácica": 0.95,
    "tontura": 0.70,
    "fraqueza": 0.80,
    "voz falhando": 0.75,
    "desmaio": 0.90,
    "confusão": 0.85,
    "formigamento": 0.75,
    "cansaço": 0.65,
    "fadiga": 0.70,
    "tosse": 0.60,
    "sufocando": 0.95,
    "coração acelerado": 0.85,
    # Inglês (para áudios do AudioSet / Kaggle)
    "shortness of breath": 0.90,
    "can't breathe": 0.95,
    "chest pain": 0.95,
    "dizziness": 0.70,
    "dizzy": 0.70,
    "weakness": 0.80,
    "fainting": 0.90,
    "fatigue": 0.70,
    "tired": 0.60,
    "cough": 0.60,
    "wheezing": 0.80,
    "headache": 0.65,
    "numbness": 0.80,
    "slurred speech": 0.90,
}

NEGATIVE_MARKERS = {
    "muita", "forte", "dor", "falhando", "falta", "pior", "angústia", "medo", "mal", "ruim",
    "pain", "worse", "bad", "hurt", "hurts", "hard to breathe", "severe", "struggling",
}
POSITIVE_MARKERS = {
    "melhor", "bem", "sem dificuldade", "sem dores", "estável", "aliviado",
    "better", "good", "fine", "no pain", "feeling well", "recovered",
}


def analyze_clinical_text(transcript: str) -> Dict[str, Any]:
    """Extrai termos críticos médicos e classifica sentimento/sofrimento clínico."""
    lower = transcript.lower()
    detected_terms: List[Dict[str, Any]] = []
    max_term_weight = 0.0

    for term, weight in CRITICAL_CLINICAL_TERMS.items():
        if term in lower:
            detected_terms.append({"term": term, "severity_weight": weight})
            max_term_weight = max(max_term_weight, weight)

    neg_hits = sum(1 for m in NEGATIVE_MARKERS if m in lower)
    pos_hits = sum(1 for m in POSITIVE_MARKERS if m in lower)

    if neg_hits > pos_hits or detected_terms:
        sentiment = "negative"
    elif pos_hits > neg_hits:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    nlp_risk_score = min(1.0, round(max_term_weight * 0.85 + (0.15 if sentiment == "negative" else 0.0), 3))

    return {
        "sentiment": sentiment,
        "critical_terms": detected_terms,
        "nlp_risk_score": nlp_risk_score,
    }
