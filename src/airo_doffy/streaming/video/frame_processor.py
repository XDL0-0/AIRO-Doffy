"""Dependency-free processing for packed camera frames before encoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...core.clocks import Clock, MonotonicClock
from ...core.errors import ModelValidationError
from ...core.types import CameraFrame, PixelFormat, ProcessedFrame


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelValidationError(f"{name} must be an integer >= 1")
    return value


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Validated geometric, color, and encoder-preparation operations."""

    crop: tuple[int, int, int, int] | None = None
    resize: tuple[int, int] | None = None
    zoom: float = 1.0
    rotation_degrees: int = 0
    output_format: PixelFormat | str | None = None
    require_even_dimensions: bool = False

    def __post_init__(self) -> None:
        if self.crop is not None:
            try:
                x, y, width, height = self.crop
            except (TypeError, ValueError) as exc:
                raise ModelValidationError("crop must contain x, y, width, height") from exc
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, int)
                or not isinstance(y, int)
                or x < 0
                or y < 0
            ):
                raise ModelValidationError("crop x and y must be non-negative integers")
            object.__setattr__(
                self,
                "crop",
                (
                    x,
                    y,
                    _positive_int(width, "crop width"),
                    _positive_int(height, "crop height"),
                ),
            )
        if self.resize is not None:
            try:
                width, height = self.resize
            except (TypeError, ValueError) as exc:
                raise ModelValidationError("resize must contain width and height") from exc
            object.__setattr__(
                self,
                "resize",
                (
                    _positive_int(width, "resize width"),
                    _positive_int(height, "resize height"),
                ),
            )
        try:
            zoom = float(self.zoom)
            output = (
                None
                if self.output_format is None
                else PixelFormat(self.output_format)
            )
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid zoom or output pixel format") from exc
        if not math.isfinite(zoom) or zoom <= 0:
            raise ModelValidationError("zoom must be positive and finite")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ModelValidationError("rotation_degrees must be 0, 90, 180, or 270")
        if not isinstance(self.require_even_dimensions, bool):
            raise ModelValidationError("require_even_dimensions must be a boolean")
        object.__setattr__(self, "zoom", zoom)
        object.__setattr__(self, "output_format", output)


def _bytes_per_pixel(pixel_format: PixelFormat) -> int:
    if pixel_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
        return 3
    if pixel_format is PixelFormat.DEPTH_U16:
        return 2
    return 1


def _dimensions(frame: CameraFrame) -> tuple[int, int]:
    return frame.shape[1], frame.shape[0]


def _crop(
    data: bytes,
    width: int,
    height: int,
    bytes_per_pixel: int,
    x: int,
    y: int,
    crop_width: int,
    crop_height: int,
) -> bytes:
    if x + crop_width > width or y + crop_height > height:
        raise ModelValidationError("crop rectangle exceeds the frame bounds")
    row_bytes = width * bytes_per_pixel
    crop_row_bytes = crop_width * bytes_per_pixel
    result = bytearray(crop_row_bytes * crop_height)
    for target_row in range(crop_height):
        source_start = (
            (y + target_row) * row_bytes
            + x * bytes_per_pixel
        )
        target_start = target_row * crop_row_bytes
        result[target_start : target_start + crop_row_bytes] = data[
            source_start : source_start + crop_row_bytes
        ]
    return bytes(result)


def _resize_nearest(
    data: bytes,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    bytes_per_pixel: int,
) -> bytes:
    if (width, height) == (target_width, target_height):
        return data
    result = bytearray(target_width * target_height * bytes_per_pixel)
    for target_y in range(target_height):
        source_y = min(height - 1, target_y * height // target_height)
        for target_x in range(target_width):
            source_x = min(width - 1, target_x * width // target_width)
            source_start = (source_y * width + source_x) * bytes_per_pixel
            target_start = (
                target_y * target_width + target_x
            ) * bytes_per_pixel
            result[target_start : target_start + bytes_per_pixel] = data[
                source_start : source_start + bytes_per_pixel
            ]
    return bytes(result)


def _rotate(
    data: bytes,
    width: int,
    height: int,
    degrees: int,
    bytes_per_pixel: int,
) -> tuple[bytes, int, int]:
    if degrees == 0:
        return data, width, height
    target_width = height if degrees in {90, 270} else width
    target_height = width if degrees in {90, 270} else height
    result = bytearray(target_width * target_height * bytes_per_pixel)
    for target_y in range(target_height):
        for target_x in range(target_width):
            if degrees == 90:
                source_x, source_y = target_y, height - 1 - target_x
            elif degrees == 180:
                source_x, source_y = width - 1 - target_x, height - 1 - target_y
            else:
                source_x, source_y = width - 1 - target_y, target_x
            source_start = (source_y * width + source_x) * bytes_per_pixel
            target_start = (
                target_y * target_width + target_x
            ) * bytes_per_pixel
            result[target_start : target_start + bytes_per_pixel] = data[
                source_start : source_start + bytes_per_pixel
            ]
    return bytes(result), target_width, target_height


def _convert_color(
    data: bytes,
    source: PixelFormat,
    target: PixelFormat,
) -> bytes:
    if source is target:
        return data
    if PixelFormat.DEPTH_U16 in {source, target}:
        raise ModelValidationError("depth frames cannot be converted as color")
    if {source, target} == {PixelFormat.RGB8, PixelFormat.BGR8}:
        result = bytearray(len(data))
        for index in range(0, len(data), 3):
            result[index : index + 3] = data[index : index + 3][::-1]
        return bytes(result)
    if source is PixelFormat.GRAY8:
        return b"".join(bytes((value, value, value)) for value in data)
    result = bytearray(len(data) // 3)
    source_is_rgb = source is PixelFormat.RGB8
    for target_index, source_index in enumerate(range(0, len(data), 3)):
        first, green, third = data[source_index : source_index + 3]
        red, blue = (first, third) if source_is_rgb else (third, first)
        result[target_index] = (77 * red + 150 * green + 29 * blue + 128) >> 8
    return bytes(result)


class PackedFrameProcessor:
    """Apply one immutable transform plan to packed RGB, gray, or depth bytes."""

    def __init__(
        self,
        transform: FrameTransform | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._transform = transform or FrameTransform()
        self._clock = clock or MonotonicClock()

    def process(self, frame: CameraFrame) -> ProcessedFrame:
        if not isinstance(frame, CameraFrame):
            raise ModelValidationError("frame must be a CameraFrame")
        transform = self._transform
        data = frame.data
        width, height = _dimensions(frame)
        pixel_format = frame.pixel_format
        bytes_per_pixel = _bytes_per_pixel(pixel_format)

        if transform.crop is not None:
            x, y, crop_width, crop_height = transform.crop
            data = _crop(
                data,
                width,
                height,
                bytes_per_pixel,
                x,
                y,
                crop_width,
                crop_height,
            )
            width, height = crop_width, crop_height

        if transform.zoom > 1.0:
            zoom_width = max(1, int(width / transform.zoom))
            zoom_height = max(1, int(height / transform.zoom))
            x = (width - zoom_width) // 2
            y = (height - zoom_height) // 2
            data = _crop(
                data,
                width,
                height,
                bytes_per_pixel,
                x,
                y,
                zoom_width,
                zoom_height,
            )
            data = _resize_nearest(
                data,
                zoom_width,
                zoom_height,
                width,
                height,
                bytes_per_pixel,
            )

        if transform.resize is not None:
            target_width, target_height = transform.resize
            data = _resize_nearest(
                data,
                width,
                height,
                target_width,
                target_height,
                bytes_per_pixel,
            )
            width, height = target_width, target_height

        data, width, height = _rotate(
            data,
            width,
            height,
            transform.rotation_degrees,
            bytes_per_pixel,
        )

        if transform.require_even_dimensions:
            even_width = width - width % 2
            even_height = height - height % 2
            if even_width < 1 or even_height < 1:
                raise ModelValidationError(
                    "frame is too small for even-dimension encoding preparation"
                )
            data = _crop(
                data,
                width,
                height,
                bytes_per_pixel,
                0,
                0,
                even_width,
                even_height,
            )
            width, height = even_width, even_height

        output_format = transform.output_format or pixel_format
        data = _convert_color(data, pixel_format, output_format)
        shape = (
            (height, width, 3)
            if output_format in {PixelFormat.RGB8, PixelFormat.BGR8}
            else (height, width)
        )
        return ProcessedFrame(
            sequence=frame.sequence,
            source_timestamp_ns=frame.source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            clock_domain=frame.clock_domain,
            stream_id=frame.stream_id,
            data=data,
            shape=shape,
            pixel_format=output_format,
            processing_timestamp_ns=self._clock.now_ns(),
        )
