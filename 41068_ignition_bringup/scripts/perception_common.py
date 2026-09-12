#!/usr/bin/env python3
"""Pure geometry and classification helpers for the perception nodes.

This module deliberately has no ROS, OpenCV, or Ultralytics imports so its
geometry can be unit-tested on a computer that does not have the full robot
software stack installed.
"""

from math import atan, radians, tan
from typing import Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


def clamp_box(box: Sequence[float], width: int, height: int) -> Box:
    """Clamp an ``(x1, y1, x2, y2)`` box to an image."""
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive.')

    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    )


def center_size_to_xyxy(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> Box:
    """Convert a center/size bounding box to corner coordinates."""
    half_width = max(0.0, float(width)) / 2.0
    half_height = max(0.0, float(height)) / 2.0
    return (
        float(center_x) - half_width,
        float(center_y) - half_height,
        float(center_x) + half_width,
        float(center_y) + half_height,
    )


def _vertical_fov(horizontal_fov_rad: float, width: int, height: int) -> float:
    """Derive vertical field of view for square pixels."""
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive.')
    return 2.0 * atan(tan(horizontal_fov_rad / 2.0) * height / width)


def _project_axis(
    coordinate: float,
    source_size: int,
    source_fov_rad: float,
    destination_size: int,
    destination_fov_rad: float,
) -> float:
    """Project one pixel coordinate between aligned pinhole cameras."""
    if source_size <= 1 or destination_size <= 1:
        raise ValueError('Image dimensions must be greater than one pixel.')

    source_normalized = 2.0 * float(coordinate) / (source_size - 1) - 1.0
    ray_tangent = source_normalized * tan(source_fov_rad / 2.0)
    destination_normalized = ray_tangent / tan(destination_fov_rad / 2.0)
    return (destination_normalized + 1.0) * (destination_size - 1) / 2.0


def project_box_between_cameras(
    box: Sequence[float],
    source_size: Tuple[int, int],
    destination_size: Tuple[int, int],
    source_hfov_deg: float,
    destination_hfov_deg: float,
    margin: float = 1.0,
    offset_pixels: Tuple[float, float] = (0.0, 0.0),
) -> Box:
    """Project a box between aligned cameras with different fields of view.

    The calculation uses pinhole ray angles, not a full-image resize. That is
    essential in this project because the 60-degree thermal image occupies
    only the central part of the 120-degree RGB image. ``offset_pixels`` is an
    optional calibration correction for small mounting/parallax errors.
    """
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    if source_hfov_deg <= 0.0 or destination_hfov_deg <= 0.0:
        raise ValueError('Camera fields of view must be positive.')
    if margin <= 0.0:
        raise ValueError('ROI margin must be positive.')

    source_hfov = radians(float(source_hfov_deg))
    destination_hfov = radians(float(destination_hfov_deg))
    source_vfov = _vertical_fov(source_hfov, source_width, source_height)
    destination_vfov = _vertical_fov(
        destination_hfov,
        destination_width,
        destination_height,
    )

    x1, y1, x2, y2 = (float(value) for value in box)
    projected = (
        _project_axis(x1, source_width, source_hfov,
                      destination_width, destination_hfov),
        _project_axis(y1, source_height, source_vfov,
                      destination_height, destination_vfov),
        _project_axis(x2, source_width, source_hfov,
                      destination_width, destination_hfov),
        _project_axis(y2, source_height, source_vfov,
                      destination_height, destination_vfov),
    )

    px1, py1, px2, py2 = projected
    center_x = (px1 + px2) / 2.0 + float(offset_pixels[0])
    center_y = (py1 + py2) / 2.0 + float(offset_pixels[1])
    half_width = abs(px2 - px1) * margin / 2.0
    half_height = abs(py2 - py1) * margin / 2.0
    return clamp_box(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        destination_width,
        destination_height,
    )


def intersection_over_smaller(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection area divided by the smaller box area."""
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)
    if smaller_area <= 0.0:
        return 0.0
    return intersection / smaller_area


def boxes_are_associated(
    thermal_roi: Sequence[float],
    rgb_box: Sequence[float],
    minimum_overlap: float,
) -> Tuple[bool, float]:
    """Decide whether an RGB detection corresponds to a projected hot ROI."""
    tx1, ty1, tx2, ty2 = (float(value) for value in thermal_roi)
    rx1, ry1, rx2, ry2 = (float(value) for value in rgb_box)
    thermal_center = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    rgb_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
    thermal_center_in_rgb = (
        rx1 <= thermal_center[0] <= rx2
        and ry1 <= thermal_center[1] <= ry2
    )
    rgb_center_in_thermal = (
        tx1 <= rgb_center[0] <= tx2
        and ty1 <= rgb_center[1] <= ty2
    )
    overlap = intersection_over_smaller(thermal_roi, rgb_box)
    return (
        thermal_center_in_rgb
        or rgb_center_in_thermal
        or overlap >= max(0.0, float(minimum_overlap)),
        overlap,
    )


def associate_detections(
    thermal_rois: Iterable[Sequence[float]],
    rgb_boxes: Sequence[Sequence[float]],
    rgb_scores: Sequence[float],
    minimum_overlap: float = 0.05,
) -> List[Optional[int]]:
    """Assign at most one RGB detection to each thermal ROI.

    The highest-confidence compatible detection wins, with overlap used as a
    secondary signal. A YOLO box cannot be assigned to two different hot ROIs.
    """
    if len(rgb_boxes) != len(rgb_scores):
        raise ValueError('RGB boxes and scores must have the same length.')

    available = set(range(len(rgb_boxes)))
    assignments: List[Optional[int]] = []
    for thermal_roi in thermal_rois:
        best_index: Optional[int] = None
        best_rank = (-1.0, -1.0)
        for index in available:
            associated, overlap = boxes_are_associated(
                thermal_roi,
                rgb_boxes[index],
                minimum_overlap,
            )
            if not associated:
                continue
            rank = (float(rgb_scores[index]), overlap)
            if rank > best_rank:
                best_rank = rank
                best_index = index
        assignments.append(best_index)
        if best_index is not None:
            available.remove(best_index)
    return assignments


def category_for_class(
    class_name: str,
    native_classes: Iterable[str],
    exotic_classes: Iterable[str],
) -> str:
    """Map a YOLO class name to ``native``, ``exotic``, or ``unknown``."""
    normalized = str(class_name).strip().lower()
    native = {str(value).strip().lower() for value in native_classes}
    exotic = {str(value).strip().lower() for value in exotic_classes}
    if normalized in native:
        return 'native'
    if normalized in exotic:
        return 'exotic'
    return 'unknown'