"""Gerador de relatórios automáticos, vídeo H.264 anotado e frames de evidência para vídeo clínico."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import PROCESSED_DATA_DIR, VIDEO_FRAMES_DIR
from src.video_analysis.openpose_analyzer import OpenPoseAnalyzer, draw_pose_skeleton
from src.video_analysis.yolo_detector import YOLOCriticalDetector


def _annotate_frame(
    frame: np.ndarray,
    pose_res: Dict[str, Any],
    yolo_res: Dict[str, Any],
    timestamp_sec: float,
    is_anomalous: bool,
) -> np.ndarray:
    """Renderiza overlay clínico completo: ROI, paciente YOLOv8, esqueleto OpenPose e HUD."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Zona Segura (ROI proporcional à resolução)
    rx1, ry1, rx2, ry2 = yolo_res["roi_pixels"]
    roi_color = (0, 60, 255) if yolo_res["roi_violation"] else (60, 190, 110)
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), roi_color, 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        "ZONA SEGURA (ROI)",
        (rx1 + 8, max(24, ry1 + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        roi_color,
        2,
        cv2.LINE_AA,
    )

    # 2. Bounding box apenas do paciente principal (evita ruído visual em cadeiras/móveis)
    patient_det = yolo_res.get("primary_patient")
    if patient_det is not None:
        bx1, by1, bx2, by2 = [int(v) for v in patient_det["bbox"]]
        box_color = (0, 0, 255) if is_anomalous else (50, 205, 50)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
        label_txt = f"Paciente ({patient_det['confidence']:.2f})"
        cv2.putText(
            annotated,
            label_txt,
            (bx1, max(25, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA,
        )

    # 3. Esqueleto Postural OpenPose / YOLOv8-Pose
    annotated = draw_pose_skeleton(annotated, pose_res)

    # 4. Barra HUD semitransparente no topo
    hud_h = 64 if w >= 800 else 52
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 20, 28), -1)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)

    status_text = "ALERTA: DESVIO DETECTADO" if is_anomalous else "POSTURA NORMAL"
    status_color = (60, 70, 255) if is_anomalous else (80, 230, 120)
    font_scale = 0.72 if w >= 800 else 0.52

    cv2.putText(
        annotated,
        f"{status_text}  |  t={timestamp_sec:.2f}s",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        status_color,
        2,
        cv2.LINE_AA,
    )

    metrics_text = (
        f"Tronco: {pose_res['trunk_tilt_deg']:.1f} deg  |  "
        f"Ombros: {pose_res['shoulder_tilt_deg']:.1f} deg  |  "
        f"Assimetria Bracos: {pose_res['arm_asymmetry_ratio'] * 100:.0f}%"
    )
    cv2.putText(
        annotated,
        metrics_text,
        (16, hud_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 if w >= 800 else 0.42,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _write_h264_video(frames_bgr: List[np.ndarray], out_path: Path, fps: float) -> Optional[str]:
    """Exporta lista de frames BGR para MP4 H.264 (compatível com navegadores/Streamlit)."""
    if not frames_bgr:
        return None
    try:
        import av

        h, w = frames_bgr[0].shape[:2]
        # H.264 exige dimensões pares
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        out_fps = max(1, int(round(fps)))

        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("libx264", rate=out_fps)
        stream.width = w_even
        stream.height = h_even
        stream.pix_fmt = "yuv420p"

        for bgr in frames_bgr:
            resized = bgr[:h_even, :w_even]
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return str(out_path)
    except Exception:
        return None


def analyze_clinical_video(
    video_path: Path,
    patient_id: str = "PAC-4001",
    sample_every_n_frames: int = 4,
    use_neural_models: bool = True,
    export_annotated_video: bool = False,
) -> Dict[str, Any]:
    """Processa vídeo clínico com OpenPose e YOLOv8, gerando timeline, frames e vídeo anotado."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    pose_analyzer = OpenPoseAnalyzer(use_neural_pose=use_neural_models)
    yolo_detector = YOLOCriticalDetector(use_neural_yolo=use_neural_models)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    anomalies: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    anomaly_candidates: List[Tuple[float, int, np.ndarray]] = []
    video_frames_out: List[np.ndarray] = []
    analyzed_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n_frames == 0:
            analyzed_count += 1
            timestamp_sec = round(frame_idx / fps, 2)
            pose_res = pose_analyzer.analyze_frame(frame)
            yolo_res = yolo_detector.detect_frame(frame)

            is_anomalous = pose_res["postural_anomaly"] or yolo_res["roi_violation"]
            reasons = list(pose_res["anomaly_reasons"])
            if yolo_res["roi_violation"]:
                reasons.append(yolo_res["violation_reason"])

            timeline.append(
                {
                    "frame": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "trunk_tilt_deg": pose_res["trunk_tilt_deg"],
                    "shoulder_tilt_deg": pose_res["shoulder_tilt_deg"],
                    "arm_asymmetry_pct": round(pose_res["arm_asymmetry_ratio"] * 100.0, 1),
                    "is_anomalous": is_anomalous,
                }
            )

            annotated_frame: Optional[np.ndarray] = None
            if is_anomalous or export_annotated_video:
                annotated_frame = _annotate_frame(frame, pose_res, yolo_res, timestamp_sec, is_anomalous)

            if export_annotated_video and annotated_frame is not None:
                video_frames_out.append(annotated_frame)

            if is_anomalous:
                anomalies.append(
                    {
                        "frame": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "trunk_tilt_deg": pose_res["trunk_tilt_deg"],
                        "shoulder_tilt_deg": pose_res["shoulder_tilt_deg"],
                        "arm_asymmetry_pct": round(pose_res["arm_asymmetry_ratio"] * 100.0, 1),
                        "reasons": reasons,
                    }
                )
                severity_rank = (
                    pose_res["trunk_tilt_deg"]
                    + pose_res["shoulder_tilt_deg"]
                    + pose_res["arm_asymmetry_ratio"] * 60.0
                    + (30.0 if yolo_res["roi_violation"] else 0.0)
                )
                if annotated_frame is not None:
                    anomaly_candidates.append((severity_rank, frame_idx, annotated_frame))

        frame_idx += 1

    cap.release()

    # Seleciona os 3 frames mais representativos (incluindo o pico máximo de anomalia)
    saved_frames: List[str] = []
    if anomaly_candidates:
        # Limpa frames antigos deste paciente para evitar confusão
        for old_img in VIDEO_FRAMES_DIR.glob(f"{patient_id}_frame_*.jpg"):
            old_img.unlink(missing_ok=True)

        ranked = sorted(anomaly_candidates, key=lambda x: x[0], reverse=True)
        chosen: List[Tuple[float, int, np.ndarray]] = []
        for cand in ranked:
            if all(abs(cand[1] - c[1]) >= sample_every_n_frames * 2 for c in chosen):
                chosen.append(cand)
            if len(chosen) >= 3:
                break
        if len(chosen) < min(3, len(ranked)):
            for cand in ranked:
                if cand not in chosen:
                    chosen.append(cand)
                if len(chosen) >= 3:
                    break
        chosen.sort(key=lambda x: x[1])

        for _, f_idx, img in chosen:
            out_img = VIDEO_FRAMES_DIR / f"{patient_id}_frame_{f_idx}.jpg"
            cv2.imwrite(str(out_img), img)
            saved_frames.append(str(out_img))

    annotated_video_path: Optional[str] = None
    if export_annotated_video and video_frames_out:
        out_vid = PROCESSED_DATA_DIR / f"annotated_{patient_id}.mp4"
        annotated_video_path = _write_h264_video(
            video_frames_out, out_vid, fps=fps / max(1, sample_every_n_frames)
        )

    anomaly_ratio = len(anomalies) / max(1, analyzed_count)
    video_risk_score = min(1.0, round(anomaly_ratio * 2.5, 3))

    report = {
        "patient_id": patient_id,
        "video_file": str(video_path),
        "total_frames": total_frames,
        "analyzed_frames": analyzed_count,
        "anomalies_detected": len(anomalies),
        "video_risk_score": video_risk_score,
        "events": anomalies,
        "timeline": timeline,
        "annotated_frames": saved_frames,
        "annotated_video_path": annotated_video_path,
    }

    report_path = PROCESSED_DATA_DIR / f"video_report_{patient_id}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
