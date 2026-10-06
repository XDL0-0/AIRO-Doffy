"""Pure image preparation shared by JPEG and WebRTC delivery."""

from __future__ import annotations

import cv2
import numpy as np


def center_zoom(
    image: np.ndarray,
    scale: float = 1.5,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Crop a centered zoom back to the original dimensions.

    The legacy behavior for scales at or below one was a same-size resize;
    retaining that detail avoids changing camera filter semantics during the
    migration.
    """

    if image.ndim < 2:
        raise ValueError("image must have height and width dimensions")
    height, width = image.shape[:2]
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("zoom scale must be finite and positive")
    new_width, new_height = int(width * scale), int(height * scale)
    if new_width <= width or new_height <= height:
        return cv2.resize(image, (width, height), interpolation=interpolation)
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    return resized[start_y : start_y + height, start_x : start_x + width]


def prepare_rgb_frame(
    frame: np.ndarray,
    *,
    zoom: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(BGR-for-transport, detached/normalized-RGB)``."""

    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB frame must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return center_zoom(bgr, zoom), rgb
