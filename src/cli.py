"""Interface de linha de comando (CLI) para demonstração do monitoramento multimodal."""
from __future__ import annotations

import argparse
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.anomaly_detection.movement_detector import analyze_patient_movement
from src.anomaly_detection.prescription_detector import analyze_prescriptions
from src.anomaly_detection.vital_signs_detector import analyze_vital_signs
from src.audio_analysis.speech_to_text import transcribe_clinical_audio
from src.audio_analysis.text_analytics import analyze_clinical_text
from src.audio_analysis.vocal_biomarkers import extract_vocal_biomarkers
from src.config import RAW_DATA_DIR
from src.data_prep.db_manager import get_patient, list_patients
from src.data_prep.real_data_downloader import prepare_all_real_datasets
from src.multimodal_fusion.alert_dispatcher import dispatch_medical_alert
from src.multimodal_fusion.fusion_engine import fuse_multimodal_risks
from src.video_analysis.video_reporter import analyze_clinical_video

console = Console()


def run_patient_pipeline(patient_id: str, use_neural_models: bool = True) -> Dict[str, Any]:
    """Executa o pipeline multimodal completo para um paciente."""
    datasets = prepare_all_real_datasets()
    patient = get_patient(patient_id)
    if not patient:
        raise ValueError(f"Paciente não encontrado: {patient_id}")

    video_path = datasets["videos"][patient["video_key"]]
    audio_path = datasets["audios"][patient["audio_key"]]
    vitals_path = RAW_DATA_DIR / patient["vitals_file"]
    presc_path = datasets["prescriptions"]

    video_report = analyze_clinical_video(video_path, patient_id=patient_id, use_neural_models=use_neural_models)
    vocal = extract_vocal_biomarkers(audio_path)
    stt = transcribe_clinical_audio(audio_path, use_whisper=use_neural_models)
    nlp = analyze_clinical_text(stt["transcript"])
    vitals = analyze_vital_signs(vitals_path)
    subject_num = int(patient_id.split("-")[1])
    presc = analyze_prescriptions(presc_path, subject_id=subject_num)
    move = analyze_patient_movement(patient_id)

    fusion = fuse_multimodal_risks(video_report, vocal, nlp, vitals, presc, move)
    summary = (
        f"Paciente {patient_id} ({patient['diagnosis_hypothesis']}) — "
        f"NEWS2={vitals['latest_news2']}, Desvios Vídeo={video_report['anomalies_detected']}, "
        f"Sentimento={nlp['sentiment']}, Prescrições Anômalas={presc['anomalies_count']}"
    )
    alert = dispatch_medical_alert(patient_id, fusion, summary)

    return {
        "patient": patient,
        "video": video_report,
        "vocal": vocal,
        "stt": stt,
        "nlp": nlp,
        "vitals": vitals,
        "prescriptions": presc,
        "movement": move,
        "fusion": fusion,
        "alert": alert,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoramento Multimodal Hospitalar - Fase 4")
    parser.add_argument("--patient", type=str, default="PAC-4001", help="ID do paciente (ex: PAC-4001)")
    parser.add_argument("--run-all", action="store_true", help="Executa para todos os pacientes")
    args = parser.parse_args()

    prepare_all_real_datasets()
    patients = list_patients() if args.run_all else [get_patient(args.patient)]

    for p in patients:
        if not p:
            continue
        res = run_patient_pipeline(p["patient_id"])
        alert = res["alert"]
        table = Table(title=f"Monitoramento Multimodal — {p['patient_id']} ({p['ward']})")
        table.add_column("Modalidade", style="cyan")
        table.add_column("Score de Risco", justify="right", style="magenta")
        for mod, score in alert["modality_scores"].items():
            table.add_row(mod, f"{score:.3f}")
        console.print(table)
        console.print(
            Panel(
                f"Severidade: [bold red]{alert['severity']}[/bold red] | "
                f"Score Integrado: {alert['risk_score']:.3f}\n{alert['summary']}",
                title="Alerta Médico em Tempo Real",
            )
        )


if __name__ == "__main__":
    main()
