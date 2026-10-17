"""
QueueSense - Visualization & HUD Overlay Engine
Renders modern semi-transparent ROI zones, motion trajectory trails,
individual dwell-time badges, and real-time operational HUD overlays.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from core.tracker import Track
from core.zone import QueueZone
from core.estimator import WaitingTimeEstimator


class QueueVisualizer:
    """
    Renders aesthetic, professional computer vision overlays for queue monitoring.
    """

    # Modern color palette (BGR format for OpenCV)
    COLOR_ZONE_CLEAR = (34, 197, 94)      # Emerald Green
    COLOR_ZONE_MODERATE = (245, 158, 11)  # Amber
    COLOR_ZONE_CONGESTED = (234, 88, 12)  # Orange
    COLOR_ZONE_CRITICAL = (239, 68, 68)   # Red
    COLOR_COUNTER = (255, 191, 0)         # Deep Sky Cyan/Amber
    COLOR_QUEUE_PERSON = (0, 215, 255)    # Gold
    COLOR_PASSBY = (120, 120, 120)        # Muted Slate
    COLOR_DARK_BG = (20, 20, 20)          # Near Black

    @classmethod
    def render(
        cls,
        frame: np.ndarray,
        active_tracks: List[Track],
        queue_zone: QueueZone,
        stats: Dict[str, Any],
        show_trajectories: bool = True,
        show_zone: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Produce fully annotated frame with zone overlays, trajectories, and badges.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw Zones (Queue ROI + Service Counter)
        if show_zone:
            annotated = cls._draw_zones(annotated, queue_zone, stats, w, h)

        # 2. Draw Trajectories (Motion Trails)
        if show_trajectories:
            cls._draw_trajectories(annotated, active_tracks)

        # 3. Draw Tracked Individuals & Badges
        for track in active_tracks:
            cls._draw_track(annotated, track, w, h)

        # 4. Draw HUD Summary Banner
        if show_hud:
            annotated = cls._draw_hud(annotated, stats, w, h)

        return annotated

    @classmethod
    def _draw_zones(cls, frame: np.ndarray, queue_zone: QueueZone, stats: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Draw semi-transparent Queue Zone and Service Counter."""
        overlay = frame.copy()

        # Pick zone color based on severity
        severity = stats.get("severity", 0)
        if severity == 0:
            zone_color = cls.COLOR_ZONE_CLEAR
        elif severity == 1:
            zone_color = cls.COLOR_ZONE_CLEAR
        elif severity == 2:
            zone_color = cls.COLOR_ZONE_MODERATE
        elif severity == 3:
            zone_color = cls.COLOR_ZONE_CONGESTED
        else:
            zone_color = cls.COLOR_ZONE_CRITICAL

        # Fill Queue Zone
        q_poly = queue_zone.get_pixel_polygon(w, h)
        cv2.fillPoly(overlay, [q_poly], zone_color)

        # Fill Counter Zone
        c_poly = queue_zone.get_counter_pixel_polygon(w, h)
        cv2.fillPoly(overlay, [c_poly], (255, 140, 0))  # Cyan/Blue

        # Blend with original for translucent glassmorphic look
        alpha = 0.18
        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw thick zone borders
        cv2.polylines(blended, [q_poly], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(blended, [c_poly], isClosed=True, color=(255, 180, 50), thickness=2, lineType=cv2.LINE_AA)

        # Counter Label
        c_centroid = np.mean(c_poly, axis=0).astype(int)
        cv2.putText(
            blended,
            "[ SERVICE COUNTER ]",
            (c_poly[0][0] + 10, c_poly[0][1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Queue Zone Label
        cv2.putText(
            blended,
            "QUEUE MONITORING ZONE",
            (q_poly[3][0] + 10, q_poly[3][1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            zone_color,
            1,
            cv2.LINE_AA,
        )

        return blended

    @classmethod
    def _draw_trajectories(cls, frame: np.ndarray, active_tracks: List[Track]) -> None:
        """Render fading motion trail for each active track."""
        for track in active_tracks:
            pts = list(track.history)
            if len(pts) < 2:
                continue

            color = cls.COLOR_QUEUE_PERSON if track.is_in_queue else cls.COLOR_PASSBY

            for i in range(1, len(pts)):
                thickness = int(np.sqrt(float(i) / len(pts)) * 3) + 1
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    @classmethod
    def _draw_track(cls, frame: np.ndarray, track: Track, w: int, h: int) -> None:
        """Render individual bounding box, ground point, and wait-time badge."""
        x1, y1, x2, y2 = track.bbox
        in_queue = track.is_in_queue

        if in_queue:
            # Color-code individual badge based on their personal dwell time
            dwell = track.total_queue_time
            if dwell < 30:
                box_color = (0, 220, 100)    # Green (Fresh arrival)
            elif dwell < 90:
                box_color = (0, 215, 255)    # Amber/Gold (Waiting)
            else:
                box_color = (30, 70, 255)    # Orange/Red (Long wait)

            # Draw sleek bracket bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
            cv2.circle(frame, track.foot_point, 4, (0, 255, 180), -1)

            # Badge text
            dwell_str = WaitingTimeEstimator.format_duration(dwell)
            if track.status == "BEING_SERVED":
                label = f"ID #{track.track_id} · SERVING"
            else:
                label = f"ID #{track.track_id} · Wait {dwell_str}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Badge pill
            bg_y1 = max(0, y1 - th - baseline - 6)
            bg_y2 = y1
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, bg_y2), box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - baseline - 2),
                font,
                font_scale,
                (10, 10, 10),
                thickness,
                cv2.LINE_AA,
            )

        else:
            # Passerby (outside queue) - Subtle dashed/faded representation
            cv2.rectangle(frame, (x1, y1), (x2, y2), cls.COLOR_PASSBY, 1, cv2.LINE_AA)
            cv2.circle(frame, track.foot_point, 3, (160, 160, 160), -1)

            label = f"#{track.track_id} Passerby"
            cv2.putText(
                frame,
                label,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                cls.COLOR_PASSBY,
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def _draw_hud(cls, frame: np.ndarray, stats: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Top HUD banner showing current queue status summary."""
        overlay = frame.copy()
        hud_h = 36
        cv2.rectangle(overlay, (0, 0), (w, hud_h), cls.COLOR_DARK_BG, -1)

        alpha = 0.75
        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Content text
        q_len = stats.get("queue_length", 0)
        est_wait = stats.get("est_wait_str", "0s")
        avg_wait = stats.get("avg_dwell_str", "0s")
        served = stats.get("served_count", 0)
        status = stats.get("congestion_label", "Clear")

        hud_text = f"QUEUE: {q_len}  |  EST WAIT: {est_wait}  |  AVG WAIT: {avg_wait}  |  SERVED: {served}"
        cv2.putText(
            blended,
            hud_text,
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Status badge on right side
        badge_text = f"[{status.upper()}]"
        severity = stats.get("severity", 0)
        if severity <= 1:
            badge_color = (50, 220, 50)
        elif severity == 2:
            badge_color = (0, 215, 255)
        elif severity == 3:
            badge_color = (0, 140, 255)
        else:
            badge_color = (40, 40, 255)

        (btw, bth), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            blended,
            badge_text,
            (w - btw - 14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        return blended
