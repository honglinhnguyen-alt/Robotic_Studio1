"""Unit tests for the pure perception helpers (no ROS needed).

Run from the package root:
    python3 -m pytest test/test_perception_fusion.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from perception_common import (  # noqa: E402
    HotRegionTracker,
    associate_detections,
    project_box_between_cameras,
    short_tag,
)


def test_neighbour_with_higher_confidence_does_not_steal_the_match():
    # Two hot regions side by side. The YOLO box of animal B is confident and
    # touches the enlarged search box of region A.
    roi_a = (85.0, 85.0, 115.0, 115.0)
    roi_b = (105.0, 85.0, 135.0, 115.0)
    box_a = (92.0, 92.0, 108.0, 108.0)      # animal A, low confidence
    box_b = (112.0, 90.0, 128.0, 110.0)     # animal B, high confidence
    assignments = associate_detections([roi_a, roi_b], [box_a, box_b], [0.55, 0.90])
    assert assignments == [0, 1]


def test_region_without_yolo_box_stays_unmatched():
    roi_a = (85.0, 85.0, 115.0, 115.0)
    roi_far = (300.0, 85.0, 310.0, 95.0)
    assignments = associate_detections([roi_a, roi_far], [(92, 92, 108, 108)], [0.8])
    assert assignments == [0, None]


def test_one_yolo_box_is_never_used_twice():
    roi_a = (90.0, 90.0, 110.0, 110.0)
    roi_b = (95.0, 90.0, 115.0, 110.0)
    assignments = associate_detections([roi_a, roi_b], [(92, 92, 112, 108)], [0.9])
    assert assignments.count(0) == 1


def test_thermal_centre_projects_to_rgb_centre():
    # Pixel centres: (320 - 1) / 2 = 159.5 and (720 - 1) / 2 = 359.5.
    roi = project_box_between_cameras(
        (149.5, 109.5, 169.5, 129.5), (320, 240), (720, 480), 60.0, 120.0)
    assert abs((roi[0] + roi[2]) / 2.0 - 359.5) < 1e-6
    assert abs((roi[1] + roi[3]) / 2.0 - 239.5) < 1e-6


def test_short_tags():
    assert short_tag('heat_12') == 'H12'
    assert short_tag('thermal_3') == 'T3'
    assert short_tag('') == '?'


def _box(cx, cy, w=10.0, h=8.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_track_needs_three_consecutive_frames():
    tracker = HotRegionTracker(persistence_frames=3)
    for frame in range(3):
        tracks = tracker.update([_box(50, 100)])
        assert len(tracks) == 1
        assert tracks[0]['confirmed'] == (frame == 2)
    assert tracks[0]['id'] == 'heat_1'


def test_ids_follow_the_region_not_the_area_order():
    tracker = HotRegionTracker(persistence_frames=1)
    first = tracker.update([_box(50, 100, 12, 10), _box(150, 100, 11, 9)])
    ids = {round((t['box'][0] + t['box'][2]) / 2): t['id'] for t in first}
    # Same regions, reported in the opposite order: their areas swapped rank.
    second = tracker.update([_box(151, 100, 12, 10), _box(49, 100, 11, 9)])
    ids_after = {round((t['box'][0] + t['box'][2]) / 2): t['id'] for t in second}
    assert ids[50] == ids_after[49]
    assert ids[150] == ids_after[151]


def test_small_moving_blob_keeps_its_id():
    tracker = HotRegionTracker(persistence_frames=1, iou_threshold=0.2)
    ids = set()
    for step in range(6):
        tracks = tracker.update([_box(50 + 6 * step, 100, 6, 6)])   # IoU with last frame = 0
        ids.add(tracks[0]['id'])
    assert ids == {'heat_1'}


def test_confirmed_track_coasts_then_expires():
    tracker = HotRegionTracker(persistence_frames=2, max_missed_frames=2)
    tracker.update([_box(50, 100)])
    tracker.update([_box(50, 100)])
    for missed in (1, 2):
        tracks = tracker.update([])
        assert len(tracks) == 1 and tracks[0]['confirmed'] and not tracks[0]['observed']
        assert tracks[0]['missed'] == missed
    assert tracker.update([]) == []
    reborn = tracker.update([_box(50, 100)])
    assert reborn[0]['id'] == 'heat_2' and not reborn[0]['confirmed']


def test_merged_regions_do_not_inherit_an_animal_id():
    tracker = HotRegionTracker(persistence_frames=1)
    first = tracker.update([_box(50, 100, 10, 8), _box(62, 100, 10, 8)])
    old_ids = {track['id'] for track in first}
    # The two hot regions touch and become one blob twice as wide.
    merged = tracker.update([_box(56, 100, 22, 8)])
    observed = [track for track in merged if track['observed']]
    assert len(observed) == 1 and observed[0]['id'] not in old_ids


def test_flickering_tentative_region_is_never_confirmed():
    tracker = HotRegionTracker(persistence_frames=3)
    for frame in range(10):
        tracks = tracker.update([_box(50, 100)] if frame % 2 == 0 else [])
        assert not any(track['confirmed'] for track in tracks)


def test_labels_never_overlap():
    cv2 = pytest.importorskip('cv2')
    from perception_overlay import LabelPlacer, draw_legend, draw_tag

    image = np.zeros((480, 720, 3), np.uint8)
    legend = draw_legend(image, [('Result', None), ('H1 deer 0.57 EXOTIC 4/5', (255, 0, 0))])
    placer = LabelPlacer(720, 480, reserved=[legend])
    rects = [legend]
    for index in range(8):
        # Eight tiny boxes packed next to each other near the horizon.
        box = (300 + 9 * index, 230, 306 + 9 * index, 238)
        rects.append(draw_tag(image, placer, f'H{index + 10}', box, (0, 200, 0)))
    for i, a in enumerate(rects):
        assert 0 <= a[0] and 0 <= a[1] and a[2] <= 720 and a[3] <= 480
        for b in rects[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    assert cv2 is not None