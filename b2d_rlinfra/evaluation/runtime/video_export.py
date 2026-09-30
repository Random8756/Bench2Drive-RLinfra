from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np


def export_bev_video(
    frames: Iterable[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 10,
) -> Optional[Path]:
    frame_list = [np.asarray(frame) for frame in frames if frame is not None]
    if not frame_list:
        return None

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    try:
        for frame in frame_list:
            if frame.shape[:2] != (height, width):
                raise ValueError("All video frames must share the same resolution")
            writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return output_path
