"""
QueueSense - Queue Zone & Service Counter Region Management
Handles polygonal and rectangular ROI definitions, point-in-polygon tests,
and filtering of passersby vs active queue members.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2


class QueueZone:
    """
    Defines a monitoring region of interest (ROI) where individuals are
    considered queued. Also defines an optional Service Counter zone.
    """

    PRESETS = {
        "Center Queue": [(0.20, 0.10), (0.80, 0.10), (0.80, 0.95), (0.20, 0.95)],
        "Left Corridor": [(0.05, 0.10), (0.50, 0.10), (0.50, 0.95), (0.05, 0.95)],
        "Right Corridor": [(0.50, 0.10), (0.95, 0.10), (0.95, 0.95), (0.50, 0.95)],
        "Full Frame": [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)],
    }

    def __init__(
        self,
        preset_name: str = "Center Queue",
        custom_norm_coords: Tuple[float, float, float, float] = None,
        counter_height_ratio: float = 0.20,
    ):
        """
        Initialize the queue monitoring zone.

        Args:
            preset_name: Key from PRESETS, or 'Custom'.
            custom_norm_coords: (xmin, ymin, xmax, ymax) in [0.0, 1.0] if preset_name is 'Custom'.
            counter_height_ratio: Upper fraction of the zone considered 'Service Counter'.
        """
        self.preset_name = preset_name
        self.counter_height_ratio = counter_height_ratio

        if preset_name == "Custom" and custom_norm_coords is not None:
            xmin, ymin, xmax, ymax = custom_norm_coords
            self.normalized_polygon = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]
        else:
            self.normalized_polygon = self.PRESETS.get(preset_name, self.PRESETS["Center Queue"])

    def get_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert normalized coordinates to absolute integer pixel vertices."""
        pts = [
            [int(nx * frame_w), int(ny * frame_h)]
            for nx, ny in self.normalized_polygon
        ]
        return np.array(pts, dtype=np.int32)

    def get_counter_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Get the front service counter sub-zone polygon (top fraction of the queue zone).
        """
        full_pts = self.get_pixel_polygon(frame_w, frame_h)
        # Top boundary is line 0-1, bottom of counter is interpolated at counter_height_ratio
        p0 = full_pts[0]
        p1 = full_pts[1]
        p2 = full_pts[2]
        p3 = full_pts[3]

        c_p2 = (
            int(p1[0] + (p2[0] - p1[0]) * self.counter_height_ratio),
            int(p1[1] + (p2[1] - p1[1]) * self.counter_height_ratio),
        )
        c_p3 = (
            int(p0[0] + (p3[0] - p0[0]) * self.counter_height_ratio),
            int(p0[1] + (p3[1] - p0[1]) * self.counter_height_ratio),
        )

        return np.array([p0, p1, c_p2, c_p3], dtype=np.int32)

    def contains_point(self, pt: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """
        Check if a given pixel coordinate is inside the queue zone.
        Uses OpenCV pointPolygonTest.
        """
        poly = self.get_pixel_polygon(frame_w, frame_h)
        dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
        return dist >= 0

    def is_in_counter(self, pt: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """Check if person is at the service counter (front of the queue)."""
        c_poly = self.get_counter_pixel_polygon(frame_w, frame_h)
        dist = cv2.pointPolygonTest(c_poly, (float(pt[0]), float(pt[1])), False)
        return dist >= 0
