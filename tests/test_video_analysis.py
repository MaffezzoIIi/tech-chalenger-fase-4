"""Testes unitários e de integração para análise de vídeo (OpenPose + YOLOv8)."""
from src.data_prep.real_data_downloader import prepare_clinical_videos
from src.video_analysis.video_reporter import analyze_clinical_video


def test_video_analysis_detects_postural_and_roi_anomalies():
    videos = prepare_clinical_videos()
    report = analyze_clinical_video(
        videos["fisioterapia"],
        patient_id="TEST-VID",
        sample_every_n_frames=5,
        use_neural_models=False,
    )
    assert report["analyzed_frames"] > 0
    assert report["anomalies_detected"] >= 1
    assert 0.0 <= report["video_risk_score"] <= 1.0
    assert len(report["annotated_frames"]) >= 1
