"""
QueueSense - Person Detection Module
Handles YOLOv8 model loading, inference for person class (COCO cls=0),
and visual annotation of detected persons.
"""

import os
# Safeguard for Windows OpenBLAS thread allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from ultralytics import YOLO


class PersonDetector:
    """
    YOLOv8-based Person Detector specifically tuned for queue and crowd monitoring.
    Filters specifically for COCO class 0 (Person).
    """

    COCO_PERSON_CLASS_ID = 0

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35, device: str = None):
        """
        Initialize the YOLOv8 person detector.

        Args:
            model_name: Path or name of YOLOv8 model weights (default 'yolov8n.pt').
            conf_threshold: Initial confidence threshold between 0.0 and 1.0.
            device: Computing device ('cpu', 'cuda', etc.). None selects best available.
        """
        self.model_name = model_name
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.model = YOLO(self.model_name)

    def set_confidence(self, conf_threshold: float) -> None:
        """Update detection confidence threshold."""
        self.conf_threshold = float(conf_threshold)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform person detection on a single BGR video frame.

        Args:
            frame: OpenCV BGR image array.

        Returns:
            List of detection dicts:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": 0,
                    "center": (cx, cy),
                    "foot_point": (cx, y2)
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        # Predict with Ultralytics YOLO, filtering strictly for class 0 (person)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.COCO_PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for xyxy, conf, cls_id in zip(xyxy_coords, confs, cls_ids):
            if int(cls_id) != self.COCO_PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            foot = (cx, y2)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "center": (cx, cy),
                "foot_point": foot,
            })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 215, 255),
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Render clean bounding boxes and tracking points on the frame.

        Args:
            frame: OpenCV BGR image array.
            detections: List of detections from detect().
            color: BGR tuple for box color (default vibrant amber/gold: (0, 215, 255)).
            show_labels: Whether to render confidence tag above box.

        Returns:
            Annotated OpenCV BGR image copy.
        """
        annotated = frame.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cx, cy = det["center"]
            fx, fy = det["foot_point"]

            # Main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Center & ground tracking dot
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 128), -1)
            cv2.circle(annotated, (fx, fy), 4, (0, 100, 255), -1)

            # Tag badge
            if show_labels:
                label = f"Person #{idx+1} {conf*100:.0f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Label background badge
                bg_y1 = max(0, y1 - label_h - baseline - 4)
                bg_y2 = y1
                cv2.rectangle(
                    annotated,
                    (x1, bg_y1),
                    (x1 + label_w + 6, bg_y2),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, y1 - baseline - 1),
                    font,
                    font_scale,
                    (15, 15, 15),
                    thickness,
                    cv2.LINE_AA,
                )

        return annotated
