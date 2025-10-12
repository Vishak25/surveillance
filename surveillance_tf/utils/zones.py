"""Zone geometry helpers."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Polygon = Sequence[Tuple[float, float]]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Return True if a point lies inside a polygon using ray casting."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def bbox_centroid_to_zone(bbox: Sequence[float], zones: Iterable[Tuple[str, Polygon]]) -> Optional[str]:
    """Return zone name containing bbox centroid, if any.

    Parameters
    ----------
    bbox: (x1, y1, x2, y2) normalised coordinates.
    zones: iterable of (name, polygon) tuples with polygon coords in [0,1].
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for name, polygon in zones:
        if point_in_polygon(cx, cy, polygon):
            return name
    return None


__all__ = ["point_in_polygon", "bbox_centroid_to_zone"]
