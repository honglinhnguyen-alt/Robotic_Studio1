#!/usr/bin/env python3
"""Pure geometry and classification helpers for the perception nodes.

This module deliberately has no ROS, OpenCV, or Ultralytics imports so its
geometry can be unit-tested on a computer that does not have the full robot
software stack installed.
"""

from math import atan, hypot, radians, tan
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
    center_tolerance: float = 0.2,
) -> Tuple[bool, float]:
    """Decide whether an RGB detection corresponds to a projected hot ROI.

    The centre of the hot region must lie inside the YOLO box, enlarged by
    ``center_tolerance`` of its size on each side (at least 3 px), or the two
    boxes must overlap by ``minimum_overlap`` of the smaller box. The hot
    region is the animal's warm body, so it sits inside that animal's YOLO
    box; a neighbouring animal's box only touches the edge of the search
    ROI, which is no longer enough.
    """
    tx1, ty1, tx2, ty2 = (float(value) for value in thermal_roi)
    rx1, ry1, rx2, ry2 = (float(value) for value in rgb_box)
    thermal_center = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    pad_x = max(3.0, float(center_tolerance) * max(0.0, rx2 - rx1))
    pad_y = max(3.0, float(center_tolerance) * max(0.0, ry2 - ry1))
    thermal_center_in_rgb = (
        rx1 - pad_x <= thermal_center[0] <= rx2 + pad_x
        and ry1 - pad_y <= thermal_center[1] <= ry2 + pad_y
    )
    overlap = intersection_over_smaller(thermal_roi, rgb_box)
    return (
        thermal_center_in_rgb or overlap >= max(0.0, float(minimum_overlap)),
        overlap,
    )


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    """Return the centre of an ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_diagonal(box: Sequence[float]) -> float:
    """Return the diagonal length of an ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def intersection_over_union(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return the IoU of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def associate_detections(
    thermal_rois: Iterable[Sequence[float]],
    rgb_boxes: Sequence[Sequence[float]],
    rgb_scores: Sequence[float],
    minimum_overlap: float = 0.05,
) -> List[Optional[int]]:
    """Assign at most one RGB detection to each thermal ROI.

    Every compatible (ROI, box) pair is ranked by geometry first: the distance
    between the ROI centre and the YOLO box centre, divided by the YOLO box
    diagonal. Pairs are then assigned globally from the best fit down, so a
    confident YOLO box on a *neighbouring* animal cannot steal the match from
    the heat region it really belongs to. YOLO confidence only breaks ties.
    A YOLO box is never assigned to two hot ROIs.
    """
    if len(rgb_boxes) != len(rgb_scores):
        raise ValueError('RGB boxes and scores must have the same length.')

    rois = [tuple(float(value) for value in roi) for roi in thermal_rois]
    pairs = []
    for roi_index, thermal_roi in enumerate(rois):
        roi_center = box_center(thermal_roi)
        for box_index, rgb_box in enumerate(rgb_boxes):
            associated, overlap = boxes_are_associated(
                thermal_roi,
                rgb_box,
                minimum_overlap,
            )
            if not associated:
                continue
            rgb_center = box_center(rgb_box)
            distance = hypot(
                roi_center[0] - rgb_center[0],
                roi_center[1] - rgb_center[1],
            )
            normalized_distance = distance / max(1.0, box_diagonal(rgb_box))
            pairs.append((
                normalized_distance,
                -overlap,
                -float(rgb_scores[box_index]),
                roi_index,
                box_index,
            ))

    pairs.sort()
    assignments: List[Optional[int]] = [None] * len(rois)
    used_boxes = set()
    for _distance, _overlap, _score, roi_index, box_index in pairs:
        if assignments[roi_index] is not None or box_index in used_boxes:
            continue
        assignments[roi_index] = box_index
        used_boxes.add(box_index)
    return assignments


def short_tag(candidate_id: str) -> str:
    """Short on-image label for a candidate ID: ``heat_12`` -> ``H12``."""
    text = str(candidate_id).strip()
    for prefix, letter in (('heat_', 'H'), ('thermal_', 'T')):
        if text.startswith(prefix) and text[len(prefix):]:
            return letter + text[len(prefix):]
    return text[:6] if text else '?'


class HotRegionTracker:
    """Give each hot region a stable ID across thermal frames.

    Regions are matched frame to frame by IoU, or by centre distance for
    small, distant blobs whose IoU drops quickly when the robot moves. A
    region must be seen in ``persistence_frames`` consecutive frames before
    it is confirmed (requirement R5). A confirmed track survives up to
    ``max_missed_frames`` missing frames, so one noisy frame does not create a
    new ID or restart the YOLO verification window.

    A match is refused when the area changes by more than ``max_area_ratio``
    between frames. That happens when two animals' hot regions merge or split,
    and without this check the ID of one animal could move to its neighbour.
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_missed_frames: int = 2,
        persistence_frames: int = 3,
        max_center_jump: float = 1.0,
        id_prefix: str = 'heat_',
        max_area_ratio: float = 1.7,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.persistence_frames = max(1, int(persistence_frames))
        self.max_center_jump = float(max_center_jump)
        self.id_prefix = str(id_prefix)
        self.max_area_ratio = max(1.0, float(max_area_ratio))
        self._tracks: List[Dict[str, object]] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []

    def update(
        self,
        boxes: Sequence[Sequence[float]],
        data: Optional[Sequence[Dict[str, object]]] = None,
    ) -> List[Dict[str, object]]:
        """Update the tracks with one frame and return every live track.

        Each returned dict has ``id``, ``box``, ``observed`` (seen in this
        frame), ``confirmed``, ``streak`` (consecutive frames seen),
        ``missed`` and ``data`` (the payload given with the box).
        """
        boxes = [tuple(float(value) for value in box) for box in boxes]
        if data is None:
            data = [{} for _ in boxes]
        if len(data) != len(boxes):
            raise ValueError('boxes and data must have the same length.')

        pairs = []
        for track_index, track in enumerate(self._tracks):
            track_box = track['box']
            track_center = box_center(track_box)
            track_area = _box_area(track_box)
            for box_index, box in enumerate(boxes):
                area = _box_area(box)
                if (
                    track_area > 0.0 and area > 0.0
                    and max(area / track_area, track_area / area) > self.max_area_ratio
                ):
                    continue
                iou = intersection_over_union(track_box, box)
                center = box_center(box)
                scale = max(4.0, box_diagonal(track_box), box_diagonal(box))
                jump = hypot(
                    track_center[0] - center[0],
                    track_center[1] - center[1],
                ) / scale
                if iou >= self.iou_threshold or jump <= self.max_center_jump:
                    pairs.append((-iou, jump, track_index, box_index))
        pairs.sort()

        matched_tracks = set()
        matched_boxes = set()
        for _iou, _jump, track_index, box_index in pairs:
            if track_index in matched_tracks or box_index in matched_boxes:
                continue
            matched_tracks.add(track_index)
            matched_boxes.add(box_index)
            track = self._tracks[track_index]
            track['box'] = boxes[box_index]
            track['data'] = dict(data[box_index])
            track['streak'] = int(track['streak']) + 1
            track['missed'] = 0
            track['observed'] = True
            if int(track['streak']) >= self.persistence_frames:
                track['confirmed'] = True

        survivors = []
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track['missed'] = int(track['missed']) + 1
                track['streak'] = 0
                track['observed'] = False
                if int(track['missed']) > self.max_missed_frames:
                    continue
                if not track['confirmed']:
                    # A tentative region must be seen in consecutive frames.
                    continue
            survivors.append(track)

        for box_index, box in enumerate(boxes):
            if box_index in matched_boxes:
                continue
            survivors.append({
                'id': f'{self.id_prefix}{self._next_id}',
                'box': box,
                'data': dict(data[box_index]),
                'streak': 1,
                'missed': 0,
                'observed': True,
                'confirmed': self.persistence_frames <= 1,
            })
            self._next_id += 1

        self._tracks = survivors
        return [
            dict(track)
            for track in sorted(
                self._tracks,
                key=lambda item: int(str(item['id'])[len(self.id_prefix):]),
            )
        ]


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