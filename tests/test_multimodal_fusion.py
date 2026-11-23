"""Testes ponta a ponta da fusão multimodal e despacho de alertas médicos."""
from src.cli import run_patient_pipeline
from src.config import AUDIT_LOG_FILE


def test_end_to_end_multimodal_fusion_and_alert():
    result = run_patient_pipeline("PAC-4001", use_neural_models=False)
    alert = result["alert"]
    assert alert["patient_id"] == "PAC-4001"
    assert alert["severity"] in {"CRÍTICO", "ALTO", "MODERADO", "NORMAL"}
    assert 0.0 <= alert["risk_score"] <= 1.0
    assert AUDIT_LOG_FILE.exists()
