"""Visualization helpers for sensor observations."""

from math import ceil
from typing import Any, Optional, Sequence

import cv2
import numpy as np

__layer__ = (5, "Evaluation")

RGB_STACK_KEYS = ("rgb",)
RGB_CAMERA_LABELS = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "BACK",
    "BACK_LEFT",
    "BACK_RIGHT",
)


def extract_sensor_visual_frame(observation: Any) -> Optional[np.ndarray]:
    """Return a display-ready RGB image for supported sensor observations."""
    if not isinstance(observation, dict):
        return None

    for key in RGB_STACK_KEYS:
        value = observation.get(key)
        if value is not None:
            return rgb_stack_to_mosaic(value)
    return None


def rgb_stack_to_mosaic(
    rgb_stack: Any,
    labels: Optional[Sequence[str]] = None,
    tile_width: int = 480,
) -> Optional[np.ndarray]:
    """Convert an RGB camera stack to a 3x2 RGB mosaic.

    The runtime RGB sensor handler stores CARLA BGRA as BGR channel order. The
    video writer path expects RGB frames, so this helper converts the stack
    before composing the mosaic.
    """
    stack = np.asarray(rgb_stack)
    if stack.ndim == 5 and stack.shape[0] == 1:
        stack = stack[0]
    if stack.ndim == 3:
        stack = stack[None, ...]
    if stack.ndim != 4:
        return None

    if stack.shape[-1] == 3:
        frames = stack
    elif stack.shape[1] == 3:
        frames = np.moveaxis(stack, 1, -1)
    else:
        return None

    if frames.shape[0] <= 0:
        return None

    frames = _as_uint8(frames)
    labels = tuple(labels or _default_labels(frames.shape[0]))

    cols = 3 if frames.shape[0] >= 3 else int(frames.shape[0])
    rows = int(ceil(frames.shape[0] / cols))
    src_h, src_w = frames.shape[1:3]
    tile_w = max(1, int(tile_width))
    tile_h = max(1, int(round(tile_w * src_h / max(src_w, 1))))

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, frame_bgr in enumerate(frames):
        row = idx // cols
        col = idx % cols
        frame_rgb = frame_bgr[..., ::-1]
        tile = cv2.resize(frame_rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        if idx < len(labels):
            tile = _draw_label(tile, labels[idx])
        y0, x0 = row * tile_h, col * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def _as_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def _default_labels(num_frames: int) -> Sequence[str]:
    if num_frames == len(RGB_CAMERA_LABELS):
        return RGB_CAMERA_LABELS
    return tuple(f"CAM_{idx}" for idx in range(num_frames))


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    tile = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_type = cv2.LINE_AA
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad_x, pad_y = 8, 6
    box_w = text_size[0] + pad_x * 2
    box_h = text_size[1] + baseline + pad_y * 2
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
    cv2.putText(
        tile,
        label,
        (pad_x, pad_y + text_size[1]),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        line_type,
    )
    return tile
