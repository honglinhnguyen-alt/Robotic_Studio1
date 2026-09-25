#!/usr/bin/env python3
"""Readable debug overlays for the thermal and RGB perception images.

Drawing rules used by both perception nodes:

* On the image itself, each heat source only gets a short tag such as ``H7``.
  Tags are placed so they never overlap each other, with a thin leader line
  when a tag has to move away from its box.
* The full result (class, confidence, native/exotic, votes) is written once
  in a legend panel in the top-left corner, one line per heat source.

This keeps the image readable even when several distant animals are only a
few pixels apart. The module only needs OpenCV and NumPy, so it can be tested
without ROS.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]
Colour = Tuple[int, int, int]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours shared by both nodes.
THERMAL_ROI_COLOUR: Colour = (0, 165, 255)   # orange: projected hot region
NATIVE_COLOUR: Colour = (60, 200, 60)        # green
EXOTIC_COLOUR: Colour = (255, 140, 30)       # blue
UNKNOWN_COLOUR: Colour = (60, 60, 255)       # red: YOLO saw no animal there
TOO_FAR_COLOUR: Colour = (0, 215, 255)       # yellow: look again (far / edge)
CONFIRMED_HEAT_COLOUR: Colour = (0, 255, 0)  # thermal image, confirmed track
TENTATIVE_HEAT_COLOUR: Colour = (0, 191, 255)
TEXT_COLOUR: Colour = (255, 255, 255)


def colour_for_status(status: str, category: str) -> Colour:
    """Colour used for one fusion result."""
    if status in ('too_far', 'partial'):
        return TOO_FAR_COLOUR
    if status != 'classified':
        return UNKNOWN_COLOUR
    if category == 'native':
        return NATIVE_COLOUR
    if category == 'exotic':
        return EXOTIC_COLOUR
    return UNKNOWN_COLOUR


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class LabelPlacer:
    """Find a free spot for each small label so labels never overlap."""

    def __init__(self, width: int, height: int, reserved: Sequence[Rect] = ()):
        self.width = int(width)
        self.height = int(height)
        self.occupied: List[Rect] = [tuple(int(v) for v in r) for r in reserved]

    def reserve(self, rect: Rect) -> None:
        self.occupied.append(tuple(int(v) for v in rect))

    def _fits(self, rect: Rect) -> bool:
        x1, y1, x2, y2 = rect
        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
            return False
        return not any(_overlaps(rect, other) for other in self.occupied)

    def place(
        self,
        size: Tuple[int, int],
        anchor: Sequence[float],
        gap: int = 2,
    ) -> Tuple[Rect, bool]:
        """Return ``(rect, adjacent)`` for a label of ``size`` near ``anchor``.

        ``adjacent`` is False when the label had to be moved away from the
        box, in which case the caller should draw a leader line.
        """
        width, height = int(size[0]), int(size[1])
        ax1, ay1, ax2, ay2 = (int(round(v)) for v in anchor)
        adjacent_spots = [
            (ax1, ay1 - gap - height),           # above, left aligned
            (ax2 - width, ay1 - gap - height),   # above, right aligned
            (ax1, ay2 + gap),                    # below, left aligned
            (ax2 - width, ay2 + gap),            # below, right aligned
            (ax2 + gap, ay1),                    # right
            (ax1 - gap - width, ay1),            # left
        ]
        for x, y in adjacent_spots:
            rect = (x, y, x + width, y + height)
            if self._fits(rect):
                self.occupied.append(rect)
                return rect, True

        # Stack further above (then below) the box, alternating sideways.
        for step in range(1, 12):
            for dx in (0, width + gap, -(width + gap)):
                for y in (
                    ay1 - gap - height - step * (height + gap),
                    ay2 + gap + step * (height + gap),
                ):
                    rect = (ax1 + dx, y, ax1 + dx + width, y + height)
                    if self._fits(rect):
                        self.occupied.append(rect)
                        return rect, False

        # Image is full: clamp to the image and accept the overlap.
        x = min(max(0, ax1), max(0, self.width - width))
        y = min(max(0, ay1 - gap - height), max(0, self.height - height))
        rect = (x, y, x + width, y + height)
        self.occupied.append(rect)
        return rect, False


def text_box_size(text: str, scale: float, thickness: int, pad: int) -> Tuple[int, int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return text_width + 2 * pad, text_height + baseline + 2 * pad, baseline


def draw_tag(
    image: np.ndarray,
    placer: LabelPlacer,
    text: str,
    anchor: Sequence[float],
    colour: Colour,
    scale: float = 0.45,
    thickness: int = 1,
    pad: int = 2,
) -> Rect:
    """Draw a small filled tag near ``anchor`` without overlapping other tags."""
    width, height, baseline = text_box_size(text, scale, thickness, pad)
    rect, adjacent = placer.place((width, height), anchor)
    x1, y1, x2, y2 = rect
    if not adjacent:
        ax1, ay1, ax2, ay2 = (int(round(v)) for v in anchor)
        anchor_point = ((ax1 + ax2) // 2, ay1 if y2 <= ay1 else ay2)
        label_point = ((x1 + x2) // 2, y2 if y2 <= ay1 else y1)
        cv2.line(image, label_point, anchor_point, colour, 1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), colour, -1)
    luminance = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    text_colour = (0, 0, 0) if luminance > 140 else TEXT_COLOUR
    cv2.putText(
        image,
        text,
        (x1 + pad, y2 - pad - baseline),
        FONT,
        scale,
        text_colour,
        thickness,
        cv2.LINE_AA,
    )
    return rect


def draw_legend(
    image: np.ndarray,
    lines: Sequence[Tuple[str, Optional[Colour]]],
    origin: Tuple[int, int] = (6, 6),
    scale: float = 0.5,
    thickness: int = 1,
    pad: int = 5,
    alpha: float = 0.6,
) -> Rect:
    """Draw a semi-transparent legend panel. ``None`` colour = plain text."""
    if not lines:
        return (0, 0, 0, 0)
    swatch = int(round(12 * scale / 0.5))
    line_gap = int(round(6 * scale / 0.5))
    sizes = [cv2.getTextSize(text, FONT, scale, thickness) for text, _ in lines]
    text_height = max(size[0][1] + size[1] for size in sizes)
    line_height = max(text_height, swatch) + line_gap
    width = max(size[0][0] for size in sizes) + swatch + 3 * pad
    height = line_height * len(lines) + pad
    x1, y1 = origin
    x2 = min(image.shape[1], x1 + width)
    y2 = min(image.shape[0], y1 + height)

    panel = image[y1:y2, x1:x2]
    if panel.size:
        dark = np.zeros_like(panel)
        image[y1:y2, x1:x2] = cv2.addWeighted(panel, 1.0 - alpha, dark, alpha, 0.0)

    for index, (text, colour) in enumerate(lines):
        top = y1 + pad + index * line_height
        text_x = x1 + pad
        if colour is not None:
            cv2.rectangle(
                image,
                (x1 + pad, top + 1),
                (x1 + pad + swatch, top + 1 + swatch),
                colour,
                -1,
            )
            text_x += swatch + pad
        cv2.putText(
            image,
            text,
            (text_x, top + max(text_height, swatch) - 2),
            FONT,
            scale,
            TEXT_COLOUR,
            thickness,
            cv2.LINE_AA,
        )
    return (x1, y1, x2, y2)