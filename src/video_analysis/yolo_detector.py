"""Detector de objetos e áreas críticas (ROIs) baseado em YOLOv8."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from src.config import SAFE_ROI_NORMALIZED, YOLO_DETECTION_MODEL


class YOLOCriticalDetector:
    """Detecta pessoas/objetos e verifica invasão ou evasão de áreas críticas (ROIs)."""

    def __init__(
        self,
        safe_roi_norm: Tuple[float, float, float, float] = SAFE_ROI_NORMALIZED,
        use_neural_yolo: bool = True,
        min_confidence: float = 0.45,
    ) -> None:
        self.safe_roi_norm = safe_roi_norm  # (x1_frac, y1_frac, x2_frac, y2_frac)
        self.min_confidence = min_confidence
        self.model = None
        if use_neural_yolo:
            try:
                from ultralytics import YOLO
                self.model = YOLO(YOLO_DETECTION_MODEL)
            except Exception:
                self.model = None

    def get_roi_pixels(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Converte a zona segura normalizada para pixels na resolução real do vídeo."""
        h, w = frame_shape[:2]
        x1 = int(self.safe_roi_norm[0] * w)
        y1 = int(self.safe_roi_norm[1] * h)
        x2 = int(self.safe_roi_norm[2] * w)
        y2 = int(self.safe_roi_norm[3] * h)
        return (x1, y1, x2, y2)

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Executa detecção no frame e verifica conformidade do paciente com a área crítica."""
        detections = self._run_detection(frame)
        x1_roi, y1_roi, x2_roi, y2_roi = self.get_roi_pixels(frame.shape)

        # Separa o paciente principal (maior bounding box da classe 'person') de objetos de fundo
        person_dets = [d for d in detections if d["label"] == "person" and d["confidence"] >= self.min_confidence]
        primary_patient = None
        if person_dets:
            primary_patient = max(
                person_dets,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            )

        roi_violation = False
        violation_reason = ""

        if primary_patient is not None:
            bx1, by1, bx2, by2 = primary_patient["bbox"]
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            if not (x1_roi <= cx <= x2_roi and y1_roi <= cy <= y2_roi):
                roi_violation = True
                violation_reason = (
                    f"Paciente fora da zona segura "
                    f"(centro=({cx:.0f},{cy:.0f}), ROI=({x1_roi}..{x2_roi}, {y1_roi}..{y2_roi}))"
                )

        return {
            "detections": detections,
            "primary_patient": primary_patient,
            "roi_pixels": (x1_roi, y1_roi, x2_roi, y2_roi),
            "roi_violation": roi_violation,
            "violation_reason": violation_reason,
        }

    def _run_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is not None:
            try:
                results = self.model.predict(frame, verbose=False)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    out: List[Dict[str, Any]] = []
                    for box in results[0].boxes:
                        conf = float(box.conf[0].item())
                        if conf < self.min_confidence:
                            continue
                        cls_id = int(box.cls[0].item())
                        label = results[0].names.get(cls_id, str(cls_id))
                        coords = [float(v) for v in box.xyxy[0].cpu().numpy()]
                        out.append({"label": label, "confidence": round(conf, 3), "bbox": coords})
                    return out
            except Exception:
                pass

        # Fallback por contorno para vídeos sintéticos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[:100, :] = 0
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:
            return []
        x, y, w, h = cv2.boundingRect(largest)
        return [{"label": "person", "confidence": 0.92, "bbox": [float(x), float(y), float(x + w), float(y + h)]}]
