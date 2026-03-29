from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class CameraFrameResult:
    ok: bool
    frame: Any | None = None
    error: str | None = None


class OpenCVCamera:
    def __init__(self, device_index: int = 0, width: int | None = None, height: int | None = None):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to use OpenCVCamera")
        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera device {self.device_index}")
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        self._cap = cap

    def read(self) -> CameraFrameResult:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            return CameraFrameResult(ok=False, error="failed to read frame")
        return CameraFrameResult(ok=True, frame=frame)

    def read_data_url(self, image_ext: str = ".jpg", quality: int = 85) -> str:
        result = self.read()
        if not result.ok or result.frame is None:
            raise RuntimeError(result.error or "failed to read frame")
        return frame_to_data_url(result.frame, image_ext=image_ext, quality=quality)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "OpenCVCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def frame_to_data_url(frame: Any, image_ext: str = ".jpg", quality: int = 85) -> str:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to encode camera frames")
    params: list[int] = []
    lower_ext = image_ext.lower()
    if lower_ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        mime = "image/jpeg"
    elif lower_ext == ".png":
        mime = "image/png"
    else:
        raise ValueError("image_ext must be .jpg, .jpeg or .png")
    ok, encoded = cv2.imencode(lower_ext, frame, params)
    if not ok:
        raise RuntimeError("failed to encode frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"
