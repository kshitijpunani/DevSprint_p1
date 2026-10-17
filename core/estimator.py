"""
QueueSense - Waiting-Time & Queue Analytics Estimator
Computes dwell times, service throughput, Little's Law waiting-time estimates,
and queue congestion classification.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from core.tracker import Track
from core.zone import QueueZone


class WaitingTimeEstimator:
    """
    Analyzes tracked pedestrians inside the designated queue zone to calculate
    dwell times, queue lengths, service throughput, and predictive wait times.
    """

    def __init__(
        self,
        default_service_sec: float = 45.0,
        alert_wait_min: float = 5.0,
        alert_queue_len: int = 8,
        alert_cooldown_sec: float = 25.0,
    ):
        """
        Initialize the estimator.

        Args:
            default_service_sec: Estimated time in seconds to serve one customer (baseline).
            alert_wait_min: Threshold in minutes above which to trigger a bottleneck alert.
            alert_queue_len: Threshold in queue count above which to trigger an alert.
            alert_cooldown_sec: Minimum seconds between logged alert occurrences (default: 25.0).
        """
        self.default_service_sec = max(5.0, float(default_service_sec))
        self.alert_wait_min = alert_wait_min
        self.alert_queue_len = alert_queue_len
        self.alert_cooldown_sec = max(5.0, float(alert_cooldown_sec))

        # Completed customers tracking
        self.served_count = 0
        self.reneged_count = 0  # Customers who left before service
        self.completed_records: List[Dict[str, Any]] = []

        # Time series history for charts: (time_sec, queue_length, est_wait_sec, avg_wait_sec)
        self.time_series_history: List[Dict[str, Any]] = []

        # Alert log
        self.alerts: List[Dict[str, Any]] = []

    def set_service_time(self, service_sec: float) -> None:
        """Update expected service duration per person."""
        self.default_service_sec = max(5.0, float(service_sec))

    def update(
        self,
        active_tracks: List[Track],
        all_tracks: Dict[int, Track],
        queue_zone: QueueZone,
        frame_w: int,
        frame_h: int,
        current_time: float,
    ) -> Dict[str, Any]:
        """
        Update state for all tracks relative to the queue and counter zones.

        Returns summary statistics dictionary for the current frame.
        """
        currently_in_queue: List[Track] = []

        for track in active_tracks:
            in_zone = queue_zone.contains_point(track.foot_point, frame_w, frame_h)
            in_counter = queue_zone.is_in_counter(track.foot_point, frame_w, frame_h)

            if in_zone:
                if not track.is_in_queue:
                    # New entrant into queue zone
                    track.is_in_queue = True
                    track.queue_entry_time = current_time
                    track.status = "WAITING"

                # Update live dwell time
                if track.queue_entry_time is not None:
                    track.total_queue_time = max(0.0, current_time - track.queue_entry_time)

                # Check if reached the counter area
                if in_counter:
                    track.status = "BEING_SERVED"
                else:
                    track.status = "WAITING"

                currently_in_queue.append(track)

            else:
                # Was in queue, now exited zone
                if track.is_in_queue:
                    dwell = track.total_queue_time
                    if track.status == "BEING_SERVED" or dwell > (self.default_service_sec * 0.5):
                        track.status = "SERVED"
                        self.served_count += 1
                    else:
                        track.status = "LEFT"
                        self.reneged_count += 1

                    self.completed_records.append({
                        "track_id": track.track_id,
                        "entry_time": track.queue_entry_time,
                        "exit_time": current_time,
                        "dwell_sec": dwell,
                        "status": track.status,
                    })

                    track.is_in_queue = False
                else:
                    track.status = "OUTSIDE"

        # Calculate statistics
        queue_len = len(currently_in_queue)
        dwell_times = [t.total_queue_time for t in currently_in_queue]

        avg_dwell_sec = float(np.mean(dwell_times)) if dwell_times else 0.0
        max_dwell_sec = float(np.max(dwell_times)) if dwell_times else 0.0

        # Effective service time (empirical if >= 3 served, else default baseline)
        if len(self.completed_records) >= 3:
            recent_completed = [r["dwell_sec"] for r in self.completed_records[-10:] if r["status"] == "SERVED"]
            if recent_completed:
                effective_service_sec = 0.5 * self.default_service_sec + 0.5 * float(np.mean(recent_completed))
            else:
                effective_service_sec = self.default_service_sec
        else:
            effective_service_sec = self.default_service_sec

        # Estimated wait time for a new arrival (Little's Law approximation: W = L / mu)
        # Each person ahead will take on average effective_service_sec
        est_wait_sec = queue_len * effective_service_sec

        # Congestion classification
        congestion_label, congestion_color, severity = self._classify_congestion(queue_len, est_wait_sec)

        # Bottleneck alert check
        is_alert = False
        if (est_wait_sec / 60.0 >= self.alert_wait_min) or (queue_len >= self.alert_queue_len):
            is_alert = True
            # Log alert every 25s to avoid duplicate spam
            if not self.alerts or (current_time - self.alerts[-1]["timestamp"] > self.alert_cooldown_sec):
                self.alerts.append({
                    "timestamp": current_time,
                    "time_str": self.format_time(current_time),
                    "queue_len": queue_len,
                    "est_wait": self.format_duration(est_wait_sec),
                    "reason": f"Queue length ({queue_len}) or Wait ({est_wait_sec/60:.1f}m) exceeded alert threshold.",
                })

        # Append to time-series (sample every 0.5s)
        if not self.time_series_history or (current_time - self.time_series_history[-1]["time_sec"] >= 0.5):
            self.time_series_history.append({
                "time_sec": round(current_time, 1),
                "queue_length": queue_len,
                "est_wait_sec": round(est_wait_sec, 1),
                "avg_wait_sec": round(avg_dwell_sec, 1),
                "served_cumulative": self.served_count,
            })

        return {
            "queue_length": queue_len,
            "avg_dwell_sec": avg_dwell_sec,
            "max_dwell_sec": max_dwell_sec,
            "est_wait_sec": est_wait_sec,
            "est_wait_str": self.format_duration(est_wait_sec),
            "avg_dwell_str": self.format_duration(avg_dwell_sec),
            "max_dwell_str": self.format_duration(max_dwell_sec),
            "congestion_label": congestion_label,
            "congestion_color": congestion_color,
            "severity": severity,
            "served_count": self.served_count,
            "is_alert": is_alert,
            "currently_in_queue": currently_in_queue,
        }

    def _classify_congestion(self, queue_len: int, est_wait_sec: float) -> Tuple[str, str, int]:
        """Classify queue into operational congestion tiers."""
        if queue_len == 0:
            return ("Clear", "#10b981", 0)  # Green
        elif queue_len <= 3 and est_wait_sec < 180:
            return ("Smooth", "#22c55e", 1)  # Emerald
        elif queue_len <= 6 and est_wait_sec < 360:
            return ("Moderate", "#f59e0b", 2)  # Amber
        elif queue_len <= 10 or est_wait_sec < 600:
            return ("Congested", "#ea580c", 3)  # Orange
        else:
            return ("Critical Bottleneck", "#ef4444", 4)  # Red

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds into readable human string: e.g. '1m 24s' or '45s'."""
        sec = int(round(seconds))
        if sec < 60:
            return f"{sec}s"
        m, s = divmod(sec, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format absolute video timestamp."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
