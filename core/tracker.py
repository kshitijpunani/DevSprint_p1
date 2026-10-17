"""
QueueSense - Multi-Object Tracker (MOT)
Tracks individuals across frames with persistent IDs, trajectory history,
and state association (IoU + Centroid distance).
"""

from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single tracked individual over time."""

    def __init__(self, track_id: int, bbox: List[int], confidence: float, timestamp: float):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.confidence = confidence
        self.first_seen = timestamp
        self.last_seen = timestamp
        
        # Centroid & foot tracking
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.centroid = (cx, cy)
        self.foot_point = (cx, bbox[3])

        # Trajectory trail (last 30 points)
        self.history: deque = deque(maxlen=30)
        self.history.append(self.foot_point)

        # Queue tracking attributes
        self.is_in_queue = False
        self.queue_entry_time: Optional[float] = None
        self.total_queue_time: float = 0.0
        self.status = "OUTSIDE"  # "WAITING", "BEING_SERVED", "SERVED", "OUTSIDE"

        # Internal tracking counters
        self.hits = 1
        self.missed_frames = 0

    def update(self, bbox: List[int], confidence: float, timestamp: float):
        """Update track with a newly matched detection."""
        # Exponential smoothing for smooth bounding box display
        alpha = 0.7
        self.bbox = [
            int(alpha * n + (1 - alpha) * o)
            for n, o in zip(bbox, self.bbox)
        ]
        self.confidence = confidence
        self.last_seen = timestamp
        self.hits += 1
        self.missed_frames = 0

        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        self.centroid = (cx, cy)
        self.foot_point = (cx, self.bbox[3])
        self.history.append(self.foot_point)

    def mark_missed(self):
        """Mark track as undetected in the current frame."""
        self.missed_frames += 1


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class QueueTracker:
    """
    Multi-object tracker associating YOLO detections across frames.
    Uses Hungarian algorithm on combined IoU and centroid distance.
    """

    def __init__(self, max_missed: int = 15, iou_threshold: float = 0.25, max_distance: float = 120.0):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

    def update(self, detections: List[Dict[str, Any]], timestamp: float) -> List[Track]:
        """
        Update tracker with current frame detections.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'.
            timestamp: Current video timestamp in seconds.

        Returns:
            List of active Track objects.
        """
        active_track_ids = list(self.tracks.keys())

        if len(active_track_ids) == 0:
            for det in detections:
                track = Track(self.next_id, det["bbox"], det["confidence"], timestamp)
                self.tracks[self.next_id] = track
                self.next_id += 1
            return list(self.tracks.values())

        if len(detections) == 0:
            # Mark all as missed
            to_delete = []
            for t_id, track in self.tracks.items():
                track.mark_missed()
                if track.missed_frames > self.max_missed:
                    to_delete.append(t_id)
            for t_id in to_delete:
                del self.tracks[t_id]
            return [t for t in self.tracks.values() if t.missed_frames == 0]

        # Compute cost matrix between existing tracks and detections
        num_tracks = len(active_track_ids)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, t_id in enumerate(active_track_ids):
            track = self.tracks[t_id]
            for j, det in enumerate(detections):
                iou = compute_iou(track.bbox, det["bbox"])
                det_cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                det_cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                dist = np.hypot(track.centroid[0] - det_cx, track.centroid[1] - det_cy)

                # Cost is low when IoU is high and distance is small
                cost = (1.0 - iou) + (dist / max(self.max_distance, 1.0)) * 0.5
                if iou < self.iou_threshold and dist > self.max_distance:
                    cost = 999.0  # Gate improbable matches

                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 2.0:  # Valid match threshold
                t_id = active_track_ids[r]
                det = detections[c]
                self.tracks[t_id].update(det["bbox"], det["confidence"], timestamp)
                assigned_tracks.add(t_id)
                assigned_dets.add(c)

        # Unmatched tracks -> increment missed count
        to_delete = []
        for t_id in active_track_ids:
            if t_id not in assigned_tracks:
                self.tracks[t_id].mark_missed()
                if self.tracks[t_id].missed_frames > self.max_missed:
                    to_delete.append(t_id)

        for t_id in to_delete:
            del self.tracks[t_id]

        # Unmatched detections -> spawn new tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                track = Track(self.next_id, det["bbox"], det["confidence"], timestamp)
                self.tracks[self.next_id] = track
                self.next_id += 1

        # Return tracks that are currently visible
        return [t for t in self.tracks.values() if t.missed_frames == 0]
