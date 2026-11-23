"""Detector de anomalias em séries temporais de sinais vitais (PhysioNet BIDMC)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import ISOLATION_FOREST_CONTAMINATION, ZSCORE_ANOMALY_THRESHOLD


def _compute_news2_score(hr: float, spo2: float, resp: float) -> int:
    """Calcula escore clínico simplificado NEWS2 para deterioração aguda."""
    score = 0
    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130 or 41 <= hr <= 50:
        score += 2
    elif 91 <= hr <= 110:
        score += 1

    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    if resp <= 8 or resp >= 25:
        score += 3
    elif 21 <= resp <= 24:
        score += 2
    elif 9 <= resp <= 11:
        score += 1
    return score


def analyze_vital_signs(csv_path: Path) -> Dict[str, Any]:
    """Analisa séries temporais reais de sinais vitais com Isolation Forest, Z-Score e NEWS2."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Mapeia colunas padrão do PhysioNet BIDMC
    hr_col = "HR" if "HR" in df.columns else df.columns[1]
    resp_col = "RESP" if "RESP" in df.columns else df.columns[2]
    spo2_col = "SpO2" if "SpO2" in df.columns else df.columns[-1]

    work = df[[hr_col, resp_col, spo2_col]].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    work.columns = ["HR", "RESP", "SpO2"]

    # Isolation Forest multivariado
    iso = IsolationForest(contamination=ISOLATION_FOREST_CONTAMINATION, random_state=42)
    iso_preds = iso.fit_predict(work[["HR", "RESP", "SpO2"]])
    work["is_iso_anomaly"] = iso_preds == -1

    # Z-Score adaptativo (EWMA) na frequência cardíaca
    ewm_mean = work["HR"].ewm(span=20, adjust=False).mean()
    ewm_std = work["HR"].ewm(span=20, adjust=False).std().fillna(1.0).replace(0.0, 1.0)
    work["hr_zscore"] = np.abs((work["HR"] - ewm_mean) / ewm_std)

    # NEWS2 no último instante e máximo observado
    news2_series = [
        _compute_news2_score(float(r.HR), float(r.SpO2), float(r.RESP))
        for r in work.itertuples(index=False)
    ]
    work["news2"] = news2_series

    anomaly_indices = work.index[
        work["is_iso_anomaly"] | (work["hr_zscore"] > ZSCORE_ANOMALY_THRESHOLD) | (work["news2"] >= 5)
    ].tolist()

    latest = work.iloc[-1]
    max_news2 = int(work["news2"].max())
    anomaly_ratio = len(anomaly_indices) / max(1, len(work))
    vitals_risk_score = min(1.0, round(0.55 * (max_news2 / 9.0) + 0.45 * min(1.0, anomaly_ratio * 3.0), 3))

    return {
        "records_analyzed": len(work),
        "anomalies_count": len(anomaly_indices),
        "anomaly_indices": anomaly_indices[:25],
        "latest_hr": round(float(latest["HR"]), 1),
        "latest_spo2": round(float(latest["SpO2"]), 1),
        "latest_resp": round(float(latest["RESP"]), 1),
        "latest_news2": int(latest["news2"]),
        "max_news2": max_news2,
        "vitals_risk_score": vitals_risk_score,
        "series": work.to_dict(orient="list"),
    }
