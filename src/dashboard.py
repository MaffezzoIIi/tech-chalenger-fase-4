"""Dashboard Web Interativo (Streamlit) para o Tech Challenge Fase 4."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection.movement_detector import analyze_patient_movement
from src.anomaly_detection.prescription_detector import analyze_prescriptions
from src.anomaly_detection.vital_signs_detector import analyze_vital_signs
from src.audio_analysis.speech_to_text import transcribe_clinical_audio
from src.audio_analysis.text_analytics import analyze_clinical_text
from src.audio_analysis.vocal_biomarkers import extract_vocal_biomarkers
from src.config import AUDIT_LOG_FILE, HUMAN_VALIDATION_DISCLAIMER, RAW_DATA_DIR
from src.data_prep.db_manager import list_patients
from src.data_prep.real_data_downloader import prepare_all_real_datasets
from src.multimodal_fusion.alert_dispatcher import dispatch_medical_alert
from src.multimodal_fusion.fusion_engine import fuse_multimodal_risks
from src.video_analysis.video_reporter import analyze_clinical_video

st.set_page_config(
    page_title="Hospital IA — Monitoramento Multimodal (Fase 4)",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Monitoramento Multimodal Preventivo Hospitalar — Fase 4")
st.caption("Arquitetura 100% Open Source Local (YOLOv8 + OpenPose + Librosa + Whisper + Isolation Forest / NEWS2)")

datasets = prepare_all_real_datasets()
patients = list_patients()
patient_map = {f"{p['patient_id']} — {p['ward']}": p for p in patients}

selected_label = st.sidebar.selectbox("Selecione o Paciente", list(patient_map.keys()))
patient = patient_map[selected_label]
patient_id = patient["patient_id"]

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Hipótese:** {patient['diagnosis_hypothesis']}\n\n"
    f"**Idade:** {patient['age']} anos\n\n"
    f"**Alergias:** {patient['allergies']}"
)

# Executa análises do paciente selecionado
video_path = datasets["videos"][patient["video_key"]]
audio_path = datasets["audios"][patient["audio_key"]]
vitals_path = RAW_DATA_DIR / patient["vitals_file"]
presc_path = datasets["prescriptions"]

video_rep = analyze_clinical_video(video_path, patient_id=patient_id)
vocal_rep = extract_vocal_biomarkers(audio_path)
stt_rep = transcribe_clinical_audio(audio_path)
nlp_rep = analyze_clinical_text(stt_rep["transcript"])
vitals_rep = analyze_vital_signs(vitals_path)
presc_rep = analyze_prescriptions(presc_path, subject_id=int(patient_id.split("-")[1]))
move_rep = analyze_patient_movement(patient_id)
fusion_rep = fuse_multimodal_risks(video_rep, vocal_rep, nlp_rep, vitals_rep, presc_rep, move_rep)

tabs = st.tabs([
    "🧠 Fusão Multimodal & Alerta",
    "🎥 Vídeo (OpenPose + YOLOv8)",
    "🎙️ Áudio & NLP (Librosa + Whisper)",
    "📈 Anomalias (PhysioNet & Leito)",
    "🚨 Auditoria (CFM / LGPD)",
])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score Multimodal", f"{fusion_rep['final_risk_score']:.2f}", f"Sinergia +{fusion_rep['synergy_bonus']:.2f}")
    col2.metric("NEWS2 Atual", vitals_rep["latest_news2"], f"Máx: {vitals_rep['max_news2']}")
    col3.metric("Desvios de Vídeo", video_rep["anomalies_detected"])
    col4.metric("Modalidades em Alerta", fusion_rep["active_modalities_in_alert"])

    df_scores = pd.DataFrame(
        {"Modalidade": list(fusion_rep["modality_scores"].keys()), "Risco": list(fusion_rep["modality_scores"].values())}
    )
    fig_bar = px.bar(df_scores, x="Modalidade", y="Risco", color="Risco", range_y=[0, 1], color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig_bar, width="stretch")

    if st.button("🚨 Disparar Alerta Médico em Tempo Real", type="primary"):
        summary = f"Alerta manual/automático para {patient_id}: Score={fusion_rep['final_risk_score']:.2f}"
        alert = dispatch_medical_alert(patient_id, fusion_rep, summary)
        st.error(f"**ALERTA {alert['severity']} EMITIDO (ID: {alert['alert_id']})** — {alert['summary']}")

with tabs[1]:
    st.subheader("Análise Postural (OpenPose) e Áreas Críticas (YOLOv8)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Frames Analisados", video_rep["analyzed_frames"])
        st.metric("Anomalias Posturais / ROI", video_rep["anomalies_detected"])
        if video_rep["events"]:
            st.dataframe(pd.DataFrame(video_rep["events"][:10]), width="stretch")
    with c2:
        if video_rep["annotated_frames"]:
            st.image(video_rep["annotated_frames"][0], caption="Frame Anotado com Detecção de Anomalia", width="stretch")

    st.markdown("---")
    st.markdown("### 📤 Testar com seu próprio vídeo clínico (.mp4)")
    uploaded_vid = st.file_uploader("Envie um vídeo de fisioterapia ou monitoramento (.mp4)", type=["mp4"])
    if uploaded_vid is not None:
        with st.spinner("Processando vídeo com OpenPose (YOLOv8-Pose) + YOLOv8 e renderizando overlay..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = Path(tmp.name)
            custom_rep = analyze_clinical_video(
                tmp_path,
                patient_id="UPLOAD",
                sample_every_n_frames=3,
                export_annotated_video=True,
            )

        um1, um2, um3 = st.columns(3)
        um1.metric("Frames Analisados", custom_rep["analyzed_frames"])
        um2.metric("Frames com Desvio", custom_rep["anomalies_detected"])
        um3.metric("Score de Risco Vídeo", f"{custom_rep['video_risk_score']:.2f}")

        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Vídeo Original**")
            st.video(str(tmp_path))
        with vc2:
            st.markdown("**Vídeo Processado (OpenPose + YOLOv8)**")
            if custom_rep.get("annotated_video_path"):
                st.video(custom_rep["annotated_video_path"])
            elif custom_rep["annotated_frames"]:
                st.image(custom_rep["annotated_frames"][0], width="stretch")

        if custom_rep.get("timeline"):
            st.markdown("**Evolução Postural ao Longo do Tempo (OpenPose)**")
            df_tl = pd.DataFrame(custom_rep["timeline"])
            fig_pose = go.Figure()
            fig_pose.add_trace(
                go.Scatter(
                    x=df_tl["timestamp_sec"],
                    y=df_tl["trunk_tilt_deg"],
                    name="Inclinação do Tronco (°)",
                    line=dict(color="crimson", width=2.5),
                )
            )
            fig_pose.add_trace(
                go.Scatter(
                    x=df_tl["timestamp_sec"],
                    y=df_tl["shoulder_tilt_deg"],
                    name="Desnivelamento de Ombros (°)",
                    line=dict(color="orange", width=2),
                )
            )
            fig_pose.add_hline(
                y=12.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Limiar de Anomalia (12°)",
            )
            fig_pose.update_layout(
                xaxis_title="Tempo (segundos)",
                yaxis_title="Ângulo (°)",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_pose, width="stretch")

        if custom_rep["annotated_frames"]:
            st.markdown("**Frames de Pico da Anomalia Detectada**")
            img_cols = st.columns(len(custom_rep["annotated_frames"]))
            for idx_col, img_path in enumerate(custom_rep["annotated_frames"]):
                img_cols[idx_col].image(
                    img_path,
                    caption=f"Evidência #{idx_col + 1} ({Path(img_path).name})",
                    width="stretch",
                )

        if custom_rep["events"]:
            st.markdown("**Relatório de Desvios Detectados**")
            st.dataframe(pd.DataFrame(custom_rep["events"]), width="stretch")
        else:
            st.success("Nenhum desvio postural ou violação de zona segura detectado neste vídeo!")

with tabs[2]:
    st.subheader("Biomarcadores Vocais (Librosa) + Transcrição (Whisper) + Termos Críticos")
    st.audio(str(audio_path))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pitch F0 (Hz)", vocal_rep["f0_mean_hz"])
    m2.metric("Jitter (%)", vocal_rep["jitter_pct"])
    m3.metric("Shimmer (%)", vocal_rep["shimmer_pct"])
    m4.metric("Pause Ratio", vocal_rep["pause_ratio"])

    st.info(f"**Transcrição ({stt_rep['engine']}):** {stt_rep['transcript']}")
    st.write(f"**Sentimento Clínico:** `{nlp_rep['sentiment']}` | **Termos Críticos:** {nlp_rep['critical_terms']}")

    st.markdown("---")
    st.markdown("### 📤 Testar com seu próprio áudio de consulta (.wav, .mp3, .ogg, .m4a) ou Microfone")
    lang_choice = st.selectbox(
        "Idioma do áudio para o Whisper",
        ["Português (pt)", "Detecção Automática", "Inglês (en)"],
        index=0,
    )
    lang_map = {"Português (pt)": "pt", "Detecção Automática": None, "Inglês (en)": "en"}

    ac_up, ac_mic = st.columns(2)
    with ac_up:
        uploaded_audio = st.file_uploader(
            "Enviar arquivo de áudio",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
        )
    with ac_mic:
        recorded_audio = st.audio_input("Ou grave uma fala de consulta pelo microfone")

    audio_source = uploaded_audio if uploaded_audio is not None else recorded_audio
    if audio_source is not None:
        suffix = Path(getattr(audio_source, "name", "recording.wav")).suffix or ".wav"
        with st.spinner("Extraindo biomarcadores vocais (Librosa) e transcrevendo com Whisper..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_a:
                tmp_a.write(audio_source.read())
                tmp_audio_path = Path(tmp_a.name)

            custom_vocal = extract_vocal_biomarkers(tmp_audio_path)
            custom_stt = transcribe_clinical_audio(
                tmp_audio_path,
                language=lang_map[lang_choice],
                use_whisper=True,
                allow_synthetic_fallback=False,
            )
            custom_nlp = analyze_clinical_text(custom_stt["transcript"])

        st.audio(str(tmp_audio_path))
        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("Duração (s)", custom_vocal["duration_sec"])
        cm2.metric("Pitch F0 (Hz)", custom_vocal["f0_mean_hz"])
        cm3.metric("Jitter (%)", custom_vocal["jitter_pct"])
        cm4.metric("Shimmer (%)", custom_vocal["shimmer_pct"])
        cm5.metric("Pause Ratio", custom_vocal["pause_ratio"])

        if custom_vocal.get("waveform_t"):
            fig_wave = go.Figure()
            fig_wave.add_trace(
                go.Scatter(
                    x=custom_vocal["waveform_t"],
                    y=custom_vocal["waveform_y"],
                    name="Amplitude Vocal",
                    line=dict(color="#1f77b4", width=1.5),
                )
            )
            fig_wave.update_layout(
                title="Forma de Onda do Áudio (Waveform)",
                xaxis_title="Tempo (s)",
                yaxis_title="Amplitude",
                height=240,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_wave, width="stretch")

        st.info(
            f"**Transcrição Real ({custom_stt['engine']} | idioma: `{custom_stt['language']}`):**\n\n"
            f"{custom_stt['transcript']}"
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Achados Acústicos (Librosa)**")
            st.metric("Score de Risco Vocal", f"{custom_vocal['vocal_risk_score']:.2f}")
            if custom_vocal["findings"]:
                for finding in custom_vocal["findings"]:
                    st.warning(finding)
            else:
                st.success("Parâmetros vocais dentro da faixa esperada.")
        with rc2:
            st.markdown("**Análise Clínica do Texto (NLP)**")
            st.metric("Score de Risco NLP", f"{custom_nlp['nlp_risk_score']:.2f}")
            st.write(f"**Sentimento Clínico:** `{custom_nlp['sentiment']}`")
            if custom_nlp["critical_terms"]:
                st.dataframe(pd.DataFrame(custom_nlp["critical_terms"]), width="stretch")
            else:
                st.success("Nenhum termo clínico crítico detectado na fala.")

with tabs[3]:
    st.subheader("Séries Temporais Reais (PhysioNet BIDMC) + Prescrições + Movimentação")
    series = vitals_rep["series"]
    fig_vs = go.Figure()
    fig_vs.add_trace(go.Scatter(y=series["HR"], name="FC (bpm)", line=dict(color="crimson")))
    fig_vs.add_trace(go.Scatter(y=series["SpO2"], name="SpO2 (%)", line=dict(color="royalblue")))
    fig_vs.add_trace(go.Scatter(y=series["RESP"], name="FR (irpm)", line=dict(color="green")))
    st.plotly_chart(fig_vs, width="stretch")

    c_p, c_m = st.columns(2)
    with c_p:
        st.markdown("#### Auditoria de Prescrições (MIMIC-III)")
        st.metric("Alertas de Prescrição", presc_rep["anomalies_count"])
        if presc_rep["alerts"]:
            st.dataframe(pd.DataFrame(presc_rep["alerts"]), width="stretch")
    with c_m:
        st.markdown("#### Padrões de Movimentação no Leito")
        st.metric("Picos de Agitação", move_rep["agitation_events"])
        st.write(move_rep["findings"] or ["Movimentação dentro do padrão esperado."])

with tabs[4]:
    st.subheader("Trilha de Auditoria (JSONL) e Conformidade Ética")
    st.warning(HUMAN_VALIDATION_DISCLAIMER)
    if AUDIT_LOG_FILE.exists():
        lines = [json.loads(line) for line in AUDIT_LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            st.dataframe(pd.DataFrame(lines[-15:]), width="stretch")
        else:
            st.info("Nenhum alerta registrado ainda.")
