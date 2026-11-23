"""Motor de fusão multimodal combinando Vídeo, Áudio/NLP, Sinais Vitais, Prescrições e Movimentação."""
from __future__ import annotations

from typing import Any, Dict


def fuse_multimodal_risks(
    video_report: Dict[str, Any],
    vocal_biomarkers: Dict[str, Any],
    nlp_analysis: Dict[str, Any],
    vitals_analysis: Dict[str, Any],
    prescription_analysis: Dict[str, Any],
    movement_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Calcula o escore integrado de risco multimodal e identifica sinergias clínicas."""
    s_video = float(video_report.get("video_risk_score", 0.0))
    s_audio = max(
        float(vocal_biomarkers.get("vocal_risk_score", 0.0)),
        float(nlp_analysis.get("nlp_risk_score", 0.0)),
    )
    s_vitals = float(vitals_analysis.get("vitals_risk_score", 0.0))
    s_presc = float(prescription_analysis.get("prescription_risk_score", 0.0))
    s_move = float(movement_analysis.get("movement_risk_score", 0.0))

    weighted_score = (
        0.22 * s_video
        + 0.23 * s_audio
        + 0.30 * s_vitals
        + 0.15 * s_presc
        + 0.10 * s_move
    )

    # Sinergia clínica: múltiplas modalidades alteradas simultaneamente amplificam o risco
    active_modalities = sum(1 for s in [s_video, s_audio, s_vitals, s_presc, s_move] if s >= 0.35)
    synergy_bonus = 0.15 if active_modalities >= 3 else (0.07 if active_modalities == 2 else 0.0)

    final_score = min(1.0, round(weighted_score + synergy_bonus, 3))

    return {
        "final_risk_score": final_score,
        "active_modalities_in_alert": active_modalities,
        "synergy_bonus": synergy_bonus,
        "modality_scores": {
            "video": round(s_video, 3),
            "audio_nlp": round(s_audio, 3),
            "vital_signs": round(s_vitals, 3),
            "prescriptions": round(s_presc, 3),
            "movement": round(s_move, 3),
        },
    }
