"""Extração de biomarcadores vocais (F0, Jitter, Shimmer, Pausas) com Librosa e fallback PyAV."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from src.config import (
    AUDIO_SAMPLE_RATE,
    JITTER_ANOMALY_THRESHOLD,
    PAUSE_RATIO_ANOMALY_THRESHOLD,
    SHIMMER_ANOMALY_THRESHOLD,
)


def _load_audio_mono(audio_path: Path, target_sr: int = AUDIO_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Carrega áudio mono (.wav, .mp3, .ogg, .m4a) via Librosa com fallback PyAV (sem exigir ffmpeg no PATH)."""
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        import av

        container = av.open(str(audio_path))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        chunks: List[np.ndarray] = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1))
        container.close()
        if not chunks:
            return np.array([], dtype=np.float32), target_sr
        return np.concatenate(chunks).astype(np.float32), target_sr


def extract_vocal_biomarkers(audio_path: Path) -> Dict[str, Any]:
    """Extrai indicadores acústicos objetivos de fadiga vocal, disartria e dispneia."""
    y, sr = _load_audio_mono(audio_path, target_sr=AUDIO_SAMPLE_RATE)
    if len(y) == 0:
        raise ValueError(f"Áudio vazio: {audio_path}")

    # Frequência fundamental (Pitch F0) via YIN
    f0 = librosa.yin(y, fmin=65.0, fmax=350.0, sr=sr)
    f0_valid = f0[(f0 > 65.0) & (f0 < 350.0)]
    f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0

    # Jitter relativo (%) - variação ciclo a ciclo da frequência fundamental
    if len(f0_valid) > 2:
        periods = 1.0 / f0_valid
        jitter_pct = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100.0)
    else:
        jitter_pct = 0.0

    # Shimmer relativo (%) - variação de amplitude RMS entre janelas
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    rms_active = rms[rms > 0.01]
    if len(rms_active) > 2:
        shimmer_pct = float(np.mean(np.abs(np.diff(rms_active))) / np.mean(rms_active) * 100.0)
    else:
        shimmer_pct = 0.0

    # Razão de pausas respiratórias (segmentos silenciosos / duração total)
    intervals = librosa.effects.split(y, top_db=22)
    active_samples = sum(end - start for start, end in intervals)
    pause_ratio = float(1.0 - (active_samples / max(1, len(y))))

    findings: List[str] = []
    if jitter_pct > JITTER_ANOMALY_THRESHOLD:
        findings.append(f"Instabilidade de pitch elevada (Jitter={jitter_pct:.2f}% — sugestivo de disartria)")
    if shimmer_pct > SHIMMER_ANOMALY_THRESHOLD:
        findings.append(f"Instabilidade de amplitude vocal (Shimmer={shimmer_pct:.2f}% — sugestivo de fadiga vocal)")
    if pause_ratio > PAUSE_RATIO_ANOMALY_THRESHOLD:
        findings.append(f"Pausas respiratórias frequentes (Pause Ratio={pause_ratio:.2f} — sugestivo de dispneia)")

    vocal_risk_score = min(
        1.0,
        round(
            0.35 * min(1.0, jitter_pct / (JITTER_ANOMALY_THRESHOLD * 2))
            + 0.35 * min(1.0, shimmer_pct / (SHIMMER_ANOMALY_THRESHOLD * 2))
            + 0.30 * min(1.0, pause_ratio / 0.6),
            3,
        ),
    )

    # Downsample da forma de onda para visualização leve no Dashboard
    n_points = min(500, len(y))
    indices = np.linspace(0, len(y) - 1, n_points, dtype=int)
    waveform_t = [round(float(i / sr), 3) for i in indices]
    waveform_y = [round(float(y[i]), 4) for i in indices]

    return {
        "duration_sec": round(len(y) / sr, 2),
        "f0_mean_hz": round(f0_mean, 2),
        "jitter_pct": round(jitter_pct, 2),
        "shimmer_pct": round(shimmer_pct, 2),
        "pause_ratio": round(pause_ratio, 3),
        "rms_mean": round(float(np.mean(rms)), 4),
        "vocal_anomaly_detected": len(findings) > 0,
        "findings": findings,
        "vocal_risk_score": vocal_risk_score,
        "waveform_t": waveform_t,
        "waveform_y": waveform_y,
    }
