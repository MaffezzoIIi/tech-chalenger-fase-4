"""Testes para biomarcadores vocais (Librosa), transcrição Whisper e NLP clínico."""
from src.audio_analysis.speech_to_text import transcribe_clinical_audio
from src.audio_analysis.text_analytics import analyze_clinical_text
from src.audio_analysis.vocal_biomarkers import extract_vocal_biomarkers
from src.data_prep.real_data_downloader import prepare_clinical_audios


def test_vocal_biomarkers_and_nlp_detect_fatigue():
    audios = prepare_clinical_audios()
    bio_anom = extract_vocal_biomarkers(audios["consulta_fadiga_disartria"])
    bio_norm = extract_vocal_biomarkers(audios["consulta_normal"])

    assert bio_anom["pause_ratio"] > bio_norm["pause_ratio"]
    assert bio_anom["vocal_risk_score"] > 0.0

    stt = transcribe_clinical_audio(audios["consulta_fadiga_disartria"], use_whisper=False)
    nlp = analyze_clinical_text(stt["transcript"])
    assert nlp["sentiment"] == "negative"
    assert len(nlp["critical_terms"]) >= 1
    assert nlp["nlp_risk_score"] > 0.5
