# src/footballplaypredictor/tracking/line_of_scrimmage.py

from __future__ import annotations
from typing import Dict, Any


def los_segment(x1: float, y1: float, x2: float, y2: float) -> Dict[str, Any]:
    """Represent a line of scrimmage as a slanted line segment."""
    return {
        "type": "segment",
        "p1": [float(x1), float(y1)],
        "p2": [float(x2), float(y2)],
    }


def signed_distance_point_to_segment(cx: float, cy: float, los: Dict[str, Any]) -> float:
    """
    Signed distance from point (cx, cy) to LOS segment.
    Useful for deciding which side of the LOS a player is on.
    """
    if los.get("type") != "segment":
        raise ValueError("LOS must be of type 'segment'")

    x1, y1 = los["p1"]
    x2, y2 = los["p2"]
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1e-6

    # unit normal (one orientation)
    nx = -dy / length
    ny = dx / length

    vx = cx - x1
    vy = cy - y1

    return vx * nx + vy * ny
