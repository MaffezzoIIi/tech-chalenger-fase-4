"""Testes para detecção de anomalias em sinais vitais PhysioNet, prescrições e movimentação."""
from src.anomaly_detection.movement_detector import analyze_patient_movement
from src.anomaly_detection.prescription_detector import analyze_prescriptions
from src.anomaly_detection.vital_signs_detector import analyze_vital_signs
from src.data_prep.real_data_downloader import (
    download_physionet_prescriptions,
    download_physionet_vitals,
)


def test_vital_signs_and_prescriptions_anomaly_detection():
    vitals_files = download_physionet_vitals()
    vitals_res = analyze_vital_signs(vitals_files[0])
    assert vitals_res["records_analyzed"] > 0
    assert 0.0 <= vitals_res["vitals_risk_score"] <= 1.0

    presc_file = download_physionet_prescriptions()
    presc_res = analyze_prescriptions(presc_file)
    assert presc_res["prescriptions_analyzed"] > 0
    assert 0.0 <= presc_res["prescription_risk_score"] <= 1.0

    move_res = analyze_patient_movement("PAC-4002")
    assert move_res["agitation_events"] >= 3
    assert move_res["movement_risk_score"] > 0.0
