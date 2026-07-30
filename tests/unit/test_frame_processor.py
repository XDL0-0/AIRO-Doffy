"""Golden tests for dependency-free packed frame processing."""

from __future__ import annotations

import unittest

from airo_doffy.core import (
    CameraFrame,
    ModelValidationError,
    PixelFormat,
)
from airo_doffy.streaming.video import (
    FrameProcessor,
    FrameTransform,
    PackedFrameProcessor,
)


class _Clock:
    def now_ns(self) -> int:
        return 999


def frame(
    data: bytes,
    shape: tuple[int, ...],
    pixel_format: PixelFormat,
) -> CameraFrame:
    return CameraFrame(
        sequence=7,
        source_timestamp_ns=100,
        receive_timestamp_ns=110,
        stream_id="camera_0",
        data=data,
        shape=shape,
        pixel_format=pixel_format,
    )


class PackedFrameProcessorTest(unittest.TestCase):
    def test_rgb_to_bgr_preserves_metadata(self) -> None:
        processor = PackedFrameProcessor(
            FrameTransform(output_format=PixelFormat.BGR8),
            clock=_Clock(),
        )
        self.assertIsInstance(processor, FrameProcessor)
        source = frame(
            b"\x01\x02\x03\x04\x05\x06",
            (1, 2, 3),
            PixelFormat.RGB8,
        )
        processed = processor.process(source)
        self.assertEqual(processed.data, b"\x03\x02\x01\x06\x05\x04")
        self.assertEqual(processed.shape, source.shape)
        self.assertEqual(processed.sequence, source.sequence)
        self.assertEqual(processed.source_timestamp_ns, source.source_timestamp_ns)
        self.assertEqual(processed.receive_timestamp_ns, source.receive_timestamp_ns)
        self.assertEqual(processed.processing_timestamp_ns, 999)

    def test_crop_resize_and_clockwise_rotation(self) -> None:
        source = frame(bytes(range(12)), (3, 4), PixelFormat.GRAY8)
        processor = PackedFrameProcessor(
            FrameTransform(
                crop=(1, 1, 2, 2),
                resize=(1, 2),
                rotation_degrees=90,
            )
        )
        processed = processor.process(source)
        self.assertEqual(processed.shape, (1, 2))
        self.assertEqual(processed.data, bytes((9, 5)))

    def test_center_zoom_uses_nearest_neighbor_and_keeps_size(self) -> None:
        source = frame(bytes(range(16)), (4, 4), PixelFormat.GRAY8)
        processed = PackedFrameProcessor(
            FrameTransform(zoom=2.0)
        ).process(source)
        self.assertEqual(processed.shape, (4, 4))
        self.assertEqual(
            processed.data,
            bytes((5, 5, 6, 6, 5, 5, 6, 6, 9, 9, 10, 10, 9, 9, 10, 10)),
        )

    def test_gray_conversion_and_even_encoding_preparation(self) -> None:
        row = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255))
        source = frame(row + row, (2, 3, 3), PixelFormat.RGB8)
        processed = PackedFrameProcessor(
            FrameTransform(
                output_format=PixelFormat.GRAY8,
                require_even_dimensions=True,
            )
        ).process(source)
        self.assertEqual(processed.shape, (2, 2))
        self.assertEqual(processed.data, bytes((77, 149, 77, 149)))

    def test_depth_geometry_and_validation(self) -> None:
        source = frame(
            b"\x01\x00\x02\x00\x03\x00\x04\x00",
            (2, 2),
            PixelFormat.DEPTH_U16,
        )
        processed = PackedFrameProcessor(
            FrameTransform(rotation_degrees=180)
        ).process(source)
        self.assertEqual(
            processed.data,
            b"\x04\x00\x03\x00\x02\x00\x01\x00",
        )
        with self.assertRaises(ModelValidationError):
            PackedFrameProcessor(
                FrameTransform(output_format=PixelFormat.RGB8)
            ).process(source)
        with self.assertRaises(ModelValidationError):
            PackedFrameProcessor(
                FrameTransform(crop=(1, 1, 2, 2))
            ).process(source)

    def test_transform_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            FrameTransform(resize=(0, 1))
        with self.assertRaises(ModelValidationError):
            FrameTransform(crop=(-1, 0, 1, 1))
        with self.assertRaises(ModelValidationError):
            FrameTransform(zoom=0)
        with self.assertRaises(ModelValidationError):
            FrameTransform(rotation_degrees=45)
        with self.assertRaises(ModelValidationError):
            FrameTransform(output_format="unknown")
        with self.assertRaises(ModelValidationError):
            FrameTransform(require_even_dimensions="yes")


if __name__ == "__main__":
    unittest.main()
