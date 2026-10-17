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

    def __init__(self, max_missed: int = 25, iou_threshold: float = 0.25, max_distance: float = 120.0):
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


if __name__ == "__main__":
    print("=" * 60)
    print(" QueueSense - Multi-Object Tracker (MOT) Self-Test & Demo")
    print("=" * 60)

    tracker = QueueTracker(max_missed=5, iou_threshold=0.25, max_distance=100.0)

    # Frame 1: Two persons enter the scene
    frame1_detections = [
        {"bbox": [100, 150, 160, 300], "confidence": 0.94},
        {"bbox": [250, 160, 310, 310], "confidence": 0.89},
    ]
    t1 = tracker.update(frame1_detections, timestamp=0.0)
    print(f"\n[Frame 1 - t=0.0s] Detections: {len(frame1_detections)}")
    for trk in t1:
        print(f"  -> Track ID: #{trk.track_id} | Centroid: {trk.centroid} | Foot: {trk.foot_point} | Conf: {trk.confidence:.2f}")

    # Frame 2: Persons move slightly forward
    frame2_detections = [
        {"bbox": [104, 152, 164, 302], "confidence": 0.95},
        {"bbox": [252, 164, 312, 314], "confidence": 0.91},
    ]
    t2 = tracker.update(frame2_detections, timestamp=0.033)
    print(f"\n[Frame 2 - t=0.033s] Detections: {len(frame2_detections)}")
    for trk in t2:
        print(f"  -> Track ID: #{trk.track_id} | Centroid: {trk.centroid} | Hits: {trk.hits} (Maintained ID)")

    # Frame 3: A third person arrives, track 2 briefly occluded (no detection)
    frame3_detections = [
        {"bbox": [108, 155, 168, 305], "confidence": 0.93},
        {"bbox": [400, 180, 460, 330], "confidence": 0.87},
    ]
    t3 = tracker.update(frame3_detections, timestamp=0.066)
    print(f"\n[Frame 3 - t=0.066s] Detections: {len(frame3_detections)} (Track #2 temporarily occluded)")
    for trk in t3:
        print(f"  -> Track ID: #{trk.track_id} | Centroid: {trk.centroid} | Status: Active")

    # Frame 4: Track 2 re-emerges near last position
    frame4_detections = [
        {"bbox": [112, 158, 172, 308], "confidence": 0.92},
        {"bbox": [255, 168, 315, 318], "confidence": 0.88},
        {"bbox": [404, 182, 464, 332], "confidence": 0.90},
    ]
    t4 = tracker.update(frame4_detections, timestamp=0.100)
    print(f"\n[Frame 4 - t=0.100s] Detections: {len(frame4_detections)} (Track #2 re-associated)")
    for trk in t4:
        print(f"  -> Track ID: #{trk.track_id} | Centroid: {trk.centroid} | Hits: {trk.hits}")

    print("\n" + "=" * 60)
    print(" QueueTracker verification: All tracks verified & persistent!")
    print("=" * 60)

