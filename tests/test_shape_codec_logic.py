"""Test cases for shape-based codec selection and dimensionality checking."""

import os
import tempfile

import numpy as np
import pytest

from robodm import FeatureType, Trajectory
from robodm.trajectory import CodecConfig


class TestShapeBasedCodecSelection:
    """Test codec selection based on data shape."""

    def test_rgb_image_codec_selection(self):
        """Test that RGB images get video codecs when compatible."""
        config = CodecConfig()

        # RGB image with even dimensions should get a video codec
        rgb_even = FeatureType(dtype="uint8", shape=(128, 128, 3))
        codec = config.get_codec_for_feature(rgb_even)
        assert (
            codec != "rawvideo"
        ), f"RGB image with even dimensions should get video codec, got {codec}"
        assert codec in [
            "libx264",
            "libx265",
            "libaom-av1",
            "ffv1",
        ], f"Got unexpected codec: {codec}"

    def test_non_rgb_shapes_use_rawvideo(self):
        """Test that non-RGB shapes always use rawvideo."""
        config = CodecConfig()

        test_cases = [
            ((128, 128), "Grayscale image"),
            ((10, ), "1D vector"),
            ((5, 10), "2D matrix"),
            ((128, 128, 1), "Single channel image"),
            ((128, 128, 4), "RGBA image"),
            ((20, 30, 5), "Multi-channel data"),
        ]

        for shape, description in test_cases:
            feature_type = FeatureType(dtype="float32", shape=shape)
            codec = config.get_codec_for_feature(feature_type)
            assert (codec == "rawvideo"
                    ), f"{description} should use rawvideo, got {codec}"

    def test_user_specified_codec_validation(self):
        """Test user-specified codec validation for RGB images."""
        # Valid user-specified codec for compatible RGB image
        config = CodecConfig(codec="libx264")
        rgb_even = FeatureType(dtype="uint8", shape=(128, 128, 3))
        codec = config.get_codec_for_feature(rgb_even)
        assert (
            codec == "libx264"
        ), f"Compatible RGB should use user-specified codec, got {codec}"

        # Invalid user-specified codec for incompatible RGB image
        config = CodecConfig(codec="libx264")
        rgb_odd = FeatureType(dtype="uint8", shape=(127, 129, 3))
        codec = config.get_codec_for_feature(rgb_odd)
        assert (
            codec == "rawvideo"
        ), f"Incompatible RGB should fall back to rawvideo, got {codec}"


class TestCodecCompatibilityValidation:
    """Test codec compatibility validation methods."""

    def test_is_valid_image_shape(self):
        """Test the is_valid_image_shape method."""
        test_cases = [
            # (shape, codec, expected_result, description)
            ((128, 128, 3), "libx264", True, "Even dimensions should work"),
            ((127, 129, 3), "libx264", False,
             "Odd dimensions should fail for H.264"),
            ((1920, 1080, 3), "libx264", True,
             "Large even dimensions should work"),
            ((2, 2, 3), "libx264", True,
             "Very small even dimensions might work"),
            (
                (128, 128),
                "libx264",
                False,
                "Non-RGB should not be valid for video codec",
            ),
            ((10, ), "libx264", False, "1D data should not be valid"),
        ]

        for shape, codec, expected, description in test_cases:
            result = CodecConfig.is_valid_image_shape(shape, codec)
            assert (
                result == expected
            ), f"{description}: shape {shape} with {codec} expected {expected}, got {result}"

    def test_is_codec_config_supported(self):
        """Test PyAV codec configuration support."""
        # These should work for most systems
        assert CodecConfig.is_codec_config_supported(128, 128, "yuv420p",
                                                     "libx264")

        # Very large dimensions might not work
        large_result = CodecConfig.is_codec_config_supported(
            10000, 10000, "yuv420p", "libx264")
        # Don't assert this as it depends on system capabilities
        print(f"Large dimensions test result: {large_result}")


class TestRoundtripData:
    """Test roundtrip encoding/decoding for various data shapes."""

    def test_different_shapes_and_types(self):
        """Test that different data shapes and types can be handled."""
        config = CodecConfig()

        test_cases = [
            # (shape, dtype, expected_codec_type)
            ((128, 128, 3), "uint8", "video"),  # RGB image
            ((100, 200, 3), "uint8", "video"),  # Different RGB size
            ((128, 128), "uint8", "rawvideo"),  # Grayscale
            ((10, ), "float32", "rawvideo"),  # Vector
            ((5, 10), "float64", "rawvideo"),  # Matrix
            ((128, 128, 1), "uint8", "rawvideo"),  # Single channel
            ((128, 128, 4), "uint8", "rawvideo"),  # RGBA
        ]

        for shape, dtype, expected_type in test_cases:
            feature_type = FeatureType(dtype=dtype, shape=shape)
            codec = config.get_codec_for_feature(feature_type)

            if expected_type == "video":
                assert (codec != "rawvideo"
                        ), f"Shape {shape} should get video codec, got {codec}"
            else:
                assert (codec == "rawvideo"
                        ), f"Shape {shape} should get rawvideo, got {codec}"

    def test_mixed_rgb_and_non_rgb_in_trajectory(self):
        """Test handling mixed RGB and non-RGB data types."""
        config = CodecConfig()

        # Simulate mixed data in a trajectory
        features = {
            "camera/rgb": FeatureType(dtype="uint8",
                                      shape=(128, 128, 3)),  # RGB
            "camera/depth": FeatureType(dtype="float32",
                                        shape=(128, 128)),  # Depth
            "robot/joint_pos": FeatureType(dtype="float32",
                                           shape=(7, )),  # Vector
            "camera/mask": FeatureType(dtype="uint8",
                                       shape=(128, 128, 1)),  # Mask
        }

        codecs = {}
        for name, feature_type in features.items():
            codecs[name] = config.get_codec_for_feature(feature_type)

        # Only RGB should get video codec
        assert codecs["camera/rgb"] != "rawvideo", "RGB should get video codec"
        assert codecs[
            "camera/depth"] == "rawvideo", "Depth should get rawvideo"
        assert (codecs["robot/joint_pos"] == "rawvideo"
                ), "Joint positions should get rawvideo"
        assert codecs["camera/mask"] == "rawvideo", "Mask should get rawvideo"


class TestPixelFormatSelection:
    """Test pixel format selection logic."""

    def test_rgb_pixel_format_selection(self):
        """Test pixel format selection for RGB data."""
        config = CodecConfig()

        rgb_type = FeatureType(dtype="uint8", shape=(128, 128, 3))

        # Test different codecs
        yuv_codecs = ["libx264", "libx265", "libaom-av1"]
        for codec in yuv_codecs:
            result = config.get_pixel_format(codec, rgb_type)
            assert (
                result == "yuv420p"
            ), f"RGB data with {codec} should get yuv420p, got {result}"

        # FFV1 uses rgb24 to avoid YUV conversion issues
        result = config.get_pixel_format("ffv1", rgb_type)
        assert result == "rgb24", f"RGB data with ffv1 should get rgb24, got {result}"

    def test_non_rgb_pixel_format_selection(self):
        """Test pixel format selection for non-RGB data."""
        config = CodecConfig()

        # Non-RGB data should not get RGB pixel formats
        grayscale_type = FeatureType(dtype="uint8", shape=(128, 128))
        vector_type = FeatureType(dtype="float32", shape=(10, ))

        # Image codecs will still return their pixel formats
        for data_type in [grayscale_type, vector_type]:
            for codec in ["libx264", "libx265", "libaom-av1"]:
                result = config.get_pixel_format(codec, data_type)
                assert (
                    result == "yuv420p"
                ), f"Image codec {codec} should return yuv420p, got {result}"

            # FFV1 returns rgb24 as default
            result = config.get_pixel_format("ffv1", data_type)
            assert result == "rgb24", f"FFV1 should return rgb24, got {result}"

    def test_rawvideo_pixel_format(self):
        """Test that rawvideo returns None for pixel format."""
        config = CodecConfig()

        rgb_type = FeatureType(dtype="uint8", shape=(128, 128, 3))
        result = config.get_pixel_format("rawvideo", rgb_type)
        assert (
            result is None
        ), f"rawvideo should return None for pixel format, got {result}"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
