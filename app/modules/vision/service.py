from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

from app.core.config import Settings

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - graceful degradation path
    cv2 = None


@dataclass(frozen=True)
class VisionAnalysis:
    source: str
    width: int | None
    height: int | None
    edge_density: float | None
    dominant_color: tuple[int, int, int] | None
    faces_detected: int | None
    object_hints: list[str]
    cv_available: bool
    warnings: list[str]


class OpenCVVisionAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(
        self,
        *,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        filename: str | None = None,
    ) -> VisionAnalysis:
        warnings: list[str] = []
        if image_bytes is None and image_url is None:
            raise ValueError("Either image_bytes or image_url must be provided.")

        source = filename or image_url or "upload"
        if image_bytes is None and image_url is not None:
            image_bytes = self._read_remote_bytes(image_url)
            source = image_url

        if cv2 is None:
            warnings.append("OpenCV is not installed; returning degraded analysis.")
            return VisionAnalysis(
                source=source,
                width=None,
                height=None,
                edge_density=None,
                dominant_color=None,
                faces_detected=None,
                object_hints=["opencv-unavailable"],
                cv_available=False,
                warnings=warnings,
            )

        assert image_bytes is not None
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image content.")

        height, width = image.shape[:2]
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale, 100, 200)
        edge_density = float(np.count_nonzero(edges) / max(width * height, 1))

        bgr_mean = image.reshape(-1, 3).mean(axis=0)
        dominant_color = (int(bgr_mean[2]), int(bgr_mean[1]), int(bgr_mean[0]))

        object_hints = ["edge-rich" if edge_density > 0.12 else "low-texture"]
        warnings.append("Face/object detection remains a placeholder in v1.")

        return VisionAnalysis(
            source=source,
            width=width,
            height=height,
            edge_density=round(edge_density, 4),
            dominant_color=dominant_color,
            faces_detected=0,
            object_hints=object_hints,
            cv_available=True,
            warnings=warnings,
        )

    def _read_remote_bytes(self, image_url: str) -> bytes:
        if not self._settings.vision_allow_remote_urls:
            raise ValueError("Remote image URLs are disabled by configuration.")

        parsed = urllib.parse.urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http/https image URLs are allowed.")

        with urllib.request.urlopen(
            image_url,
            timeout=self._settings.vision_remote_timeout_seconds,
        ) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                raise ValueError("Remote URL did not return an image content type.")
            return response.read()
