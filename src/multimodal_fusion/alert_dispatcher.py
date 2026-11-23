"""Despachante de alertas médicos em tempo real com trilha de auditoria JSONL e SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict
import uuid

from src.config import AUDIT_LOG_FILE, HUMAN_VALIDATION_DISCLAIMER
from src.data_prep.db_manager import log_alert_to_db


def dispatch_medical_alert(
    patient_id: str,
    fusion_result: Dict[str, Any],
    summary: str,
) -> Dict[str, Any]:
    """Classifica severidade, registra trilha de auditoria e retorna alerta estruturado."""
    score = float(fusion_result["final_risk_score"])
    if score >= 0.70:
        severity = "CRÍTICO"
    elif score >= 0.45:
        severity = "ALTO"
    elif score >= 0.25:
        severity = "MODERADO"
    else:
        severity = "NORMAL"

    timestamp = datetime.now(timezone.utc).isoformat()
    alert_payload = {
        "alert_id": str(uuid.uuid4())[:8],
        "timestamp_utc": timestamp,
        "patient_id": patient_id,
        "severity": severity,
        "risk_score": score,
        "modality_scores": fusion_result["modality_scores"],
        "summary": summary,
        "disclaimer": HUMAN_VALIDATION_DISCLAIMER,
    }

    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(alert_payload, ensure_ascii=False) + "\n")

    log_alert_to_db(patient_id, severity, score, summary, timestamp)
    return alert_payload
