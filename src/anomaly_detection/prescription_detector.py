"""Detector de anomalias na evolução de prescrições médicas (MIMIC-III / Hospital)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

HIGH_RISK_DRUGS = {"Noradrenalina", "Midazolam", "Morfina", "Insulina", "Heparina", "Norepinephrine"}


def analyze_prescriptions(csv_path: Path, subject_id: int | None = None) -> Dict[str, Any]:
    """Detecta alterações inesperadas de dose e riscos em prescrições médicas."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]

    if subject_id is not None and "SUBJECT_ID" in df.columns:
        sub = df[df["SUBJECT_ID"] == subject_id]
        if not sub.empty:
            df = sub

    df["DOSE_NUM"] = pd.to_numeric(df.get("DOSE_VAL_RX", 0), errors="coerce")
    alerts: List[Dict[str, Any]] = []

    if "DRUG" in df.columns:
        for drug, grp in df.groupby("DRUG"):
            doses = grp["DOSE_NUM"].dropna().tolist()
            if len(doses) >= 2:
                prev_dose, curr_dose = doses[-2], doses[-1]
                if prev_dose > 0 and (curr_dose / prev_dose) >= 2.0:
                    alerts.append(
                        {
                            "drug": str(drug),
                            "type": "Escalonamento abrupto de dose",
                            "detail": f"{prev_dose} -> {curr_dose} (+{((curr_dose/prev_dose)-1)*100:.0f}%)",
                            "severity": "ALTA" if str(drug) in HIGH_RISK_DRUGS else "MODERADA",
                        }
                    )

    risk_score = min(1.0, round(len(alerts) * 0.45, 3))
    return {
        "prescriptions_analyzed": len(df),
        "anomalies_count": len(alerts),
        "alerts": alerts,
        "prescription_risk_score": risk_score,
    }
