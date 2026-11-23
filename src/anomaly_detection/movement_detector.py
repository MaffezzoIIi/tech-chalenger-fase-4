"""Detector de anomalias em padrões de movimentação do paciente durante a internação."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def analyze_patient_movement(patient_id: str) -> Dict[str, Any]:
    """Analisa telemetria de movimentação no leito (agitação, risco de queda, imobilidade)."""
    rng = np.random.default_rng(abs(hash(patient_id)) % (2**31))
    n = 60
    accel = np.abs(rng.normal(0.15, 0.05, n))

    # Simula perfil clínico por paciente
    if patient_id == "PAC-4002":
        accel[-12:] += np.linspace(0.4, 1.4, 12)  # Agitação psicomotora / risco de queda
    elif patient_id == "PAC-4001":
        accel[-25:] = 0.01  # Imobilidade prolongada / déficit motor

    agitation_events = int(np.sum(accel > 0.85))
    immobility_streak = int(np.sum(accel < 0.03))

    findings: List[str] = []
    if agitation_events >= 3:
        findings.append(f"Agitação psicomotora / risco de queda ({agitation_events} picos > 0.85g)")
    if immobility_streak >= 20:
        findings.append(f"Imobilidade prolongada detectada ({immobility_streak} janelas sem reposicionamento)")

    movement_risk_score = min(1.0, round(0.15 * agitation_events + (0.45 if immobility_streak >= 20 else 0.0), 3))

    return {
        "samples": n,
        "agitation_events": agitation_events,
        "immobility_streak": immobility_streak,
        "findings": findings,
        "movement_risk_score": movement_risk_score,
        "accel_series": [round(float(v), 3) for v in accel],
    }
