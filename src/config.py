"""Configurações globais do Sistema de Monitoramento Multimodal Hospitalar (Fase 4)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Diretórios de Dados e Artefatos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIDEO_FRAMES_DIR = PROCESSED_DATA_DIR / "video_frames"
DATABASE_PATH = DATA_DIR / "hospital_multimodal.sqlite"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "multimodal_alerts.jsonl"

# Garantir criação automática dos diretórios
for _path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VIDEO_FRAMES_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    _path.mkdir(parents=True, exist_ok=True)

# Configurações de Visão Computacional (OpenPose / YOLOv8)
YOLO_DETECTION_MODEL = str(MODELS_DIR / "yolov8n.pt")
YOLO_POSE_MODEL = str(MODELS_DIR / "yolov8n-pose.pt")
POSE_CONFIDENCE_THRESHOLD = 0.35
TRUNK_TILT_ANOMALY_DEG = 12.0
SHOULDER_TILT_ANOMALY_DEG = 10.0
ARM_ASYMMETRY_RATIO_THRESHOLD = 0.25
ARM_ASYMMETRY_THRESHOLD_PX = 45.0
SAFE_ROI_NORMALIZED = (0.15, 0.11, 0.85, 0.96)  # (x1_frac, y1_frac, x2_frac, y2_frac)

# Configurações de Áudio e NLP Open Source
WHISPER_MODEL_SIZE = "tiny"
AUDIO_SAMPLE_RATE = 16000
JITTER_ANOMALY_THRESHOLD = 2.0      # %
SHIMMER_ANOMALY_THRESHOLD = 6.0     # %
PAUSE_RATIO_ANOMALY_THRESHOLD = 0.35

# Configurações de Detecção de Anomalias
ISOLATION_FOREST_CONTAMINATION = 0.08
ZSCORE_ANOMALY_THRESHOLD = 2.5

# Disclaimer Ético e Regulatório (CFM / LGPD)
HUMAN_VALIDATION_DISCLAIMER = (
    "AVISO DE SEGURANÇA MÉDICA (CFM / LGPD): Este sistema atua exclusivamente como suporte "
    "preventivo à decisão clínica por meio de monitoramento multimodal local (On-Premise). "
    "Alertas automáticos NÃO substituem a avaliação médica presencial e devem ser validados "
    "pela equipe assistencial responsável."
)
