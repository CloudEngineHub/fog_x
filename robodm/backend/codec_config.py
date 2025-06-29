import logging
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import av

from robodm.feature import FeatureType

logger = logging.getLogger(__name__)


class CodecConfig:
    """Configuration class for video codec settings with feature-specific codec mapping."""

    @staticmethod
    def get_supported_pixel_formats(codec_name: str) -> List[str]:
        """Get list of supported pixel formats for a codec."""
        try:
            import av

            codec = av.codec.Codec(codec_name, "w")
            if codec.video_formats:
                return [vf.name for vf in codec.video_formats]
            return []
        except Exception:
            return []

    @staticmethod
    def is_codec_config_supported(width: int,
                                  height: int,
                                  pix_fmt: str = "yuv420p",
                                  codec_name: str = "libx264") -> bool:
        """Check if a specific width/height/pixel format combination is supported by codec."""
        try:
            cc = av.codec.CodecContext.create(codec_name, "w")
            cc.width = width  # type: ignore[attr-defined]
            cc.height = height  # type: ignore[attr-defined]
            cc.pix_fmt = pix_fmt  # type: ignore[attr-defined]
            cc.time_base = Fraction(1, 30)
            cc.open(strict=True)
            # Note: CodecContext doesn't have a close() method in newer PyAV versions
            return True
        except Exception as e:
            logger.debug(f"Codec config validation failed: {e}")
            return False

    @staticmethod
    def is_valid_image_shape(shape: Tuple[int, ...],
                             codec_name: str = "libx264") -> bool:
        """Check if a shape can be treated as an RGB image for the given codec."""
        # Only accept RGB shapes (H, W, 3)
        if len(shape) != 3 or shape[2] != 3:
            return False

        height, width = shape[0], shape[1]

        # Check minimum reasonable image size
        if height < 1 or width < 1:
            return False

        # Check codec-specific constraints
        if codec_name in ["libx264", "libx265"]:
            # H.264/H.265 require even dimensions
            if height % 2 != 0 or width % 2 != 0:
                return False
        elif codec_name in ["libaom-av1"]:
            # AV1 also typically requires even dimensions for yuv420p
            if height % 2 != 0 or width % 2 != 0:
                return False
        elif codec_name == "ffv1":
            # FFV1 can handle odd dimensions but requires minimal size
            if height < 2 or width < 2:
                return False

        # Test if the codec actually supports this resolution
        # For FFV1, test with rgb24 instead of yuv420p
        if codec_name == "ffv1":
            return CodecConfig.is_codec_config_supported(
                width, height, "rgb24", codec_name)
        else:
            return CodecConfig.is_codec_config_supported(
                width, height, "yuv420p", codec_name)

    @staticmethod
    def is_image_codec(codec_name: str) -> bool:
        """Check if a codec is an image/video codec."""
        return codec_name in {"libx264", "libx265", "libaom-av1", "ffv1"}

    @staticmethod
    def is_raw_data_codec(codec_name: str) -> bool:
        """Check if a codec is for raw/non-image data."""
        return codec_name.startswith("rawvideo")

    # Image codec configurations (use actual codec for container)
    IMAGE_CODEC_CONFIGS: Dict[str, Dict[str, Any]] = {
        "libx264": {
            "container_codec": "libx264",  # Use actual codec for container
            "pixel_format": "yuv420p",
            "options": {
                "crf": "23",
                "preset": "medium"
            },
        },
        "libx265": {
            "container_codec": "libx265",  # Use actual codec for container
            "pixel_format": "yuv420p",
            "options": {
                "crf": "28",
                "preset": "medium"
            },
        },
        "libaom-av1": {
            "container_codec": "libaom-av1",  # Use actual codec for container
            "pixel_format": "yuv420p",
            "options": {
                "g": "2",
                "crf": "30"
            },
        },
        "ffv1": {
            "container_codec": "ffv1",  # Use actual codec for container
            "pixel_format":
            "yuv420p",  # Default, will be adjusted based on content
            "options": {},
        },
    }

    # Raw data codec configurations (always use rawvideo container)
    RAW_DATA_CODEC_CONFIGS: Dict[str, Dict[str, Any]] = {
        "rawvideo": {
            "container_codec": "rawvideo",  # Always rawvideo for container
            "internal_codec": "pickle_raw",  # Default internal implementation
            "options": {},
        },
        "rawvideo_pickle": {
            "container_codec": "rawvideo",  # Always rawvideo for container
            "internal_codec": "pickle_raw",
            "options": {},
        },
        "rawvideo_pyarrow": {
            "container_codec": "rawvideo",  # Always rawvideo for container
            "internal_codec": "pyarrow_batch",
            "options": {
                "batch_size": 100,
                "compression": "snappy"
            },
        },
    }

    # Backward compatibility: Combined codec configs
    @property
    def CODEC_CONFIGS(self) -> Dict[str, Dict[str, Any]]:
        """Legacy CODEC_CONFIGS property for backward compatibility."""
        configs = {}

        # Add image codecs
        for codec_name, config in self.IMAGE_CODEC_CONFIGS.items():
            configs[codec_name] = {
                "pixel_format": config.get("pixel_format"),
                "options": config.get("options", {}),
                "container_codec": config.get("container_codec"),
            }

        # Add raw data codecs
        for codec_name, config in self.RAW_DATA_CODEC_CONFIGS.items():
            configs[codec_name] = {
                "pixel_format": None,  # Raw data doesn't use pixel formats
                "options": config.get("options", {}),
                "raw_codec": config.get("internal_codec"),
                "container_codec": config.get("container_codec"),
            }

        return configs

    def __init__(
        self,
        codec: Union[str, Dict[str, str]] = "auto",
        options: Optional[Dict[str, Any]] = None,
        video_codec: Optional[str] = None,
        raw_codec: Optional[str] = None,
    ):
        """
        Initialize codec configuration.

        Args:
            codec: Either a default codec string ("auto", "rawvideo", etc.) or
                   a dictionary mapping feature names to specific codecs {feature_name: codec}
            options: Additional codec-specific options
            video_codec: Specific codec to use for video/image features (RGB images)
            raw_codec: Specific codec to use for raw data features (non-RGB data)
        """
        if isinstance(codec, dict):
            # Feature-specific codec mapping
            self.feature_codecs = codec
            self.codec = "auto"  # Default for unmapped features
        else:
            # Single codec for all features
            self.codec = codec
            self.feature_codecs = {}

        # Store specific video and raw codec preferences
        self.video_codec = video_codec
        self.raw_codec = raw_codec

        # Separate custom options by codec type
        self.custom_options = options or {}
        self.video_custom_options = {}
        self.raw_custom_options = {}

        # Separate options based on known option names
        if self.custom_options:
            # Video codec option names
            video_option_names = {
                "crf",
                "preset",
                "g",
                "profile",
                "level",
                "tune",
                "x264-params",
                "x265-params",
            }
            # Raw codec option names
            raw_option_names = {"batch_size", "compression", "algorithm"}

            for key, value in self.custom_options.items():
                if key in video_option_names:
                    self.video_custom_options[key] = value
                elif key in raw_option_names:
                    self.raw_custom_options[key] = value
                # If unknown, don't assign to either (safer than guessing)

        # Validate all specified codecs
        all_codecs = set([self.codec])
        if self.video_codec:
            all_codecs.add(self.video_codec)
        if self.raw_codec:
            all_codecs.add(self.raw_codec)
        all_codecs.update(self.feature_codecs.values())

        for codec_name in all_codecs:
            if codec_name not in ["auto"
                                  ] and not self._is_valid_codec(codec_name):
                available_codecs = list(
                    self.IMAGE_CODEC_CONFIGS.keys()) + list(
                        self.RAW_DATA_CODEC_CONFIGS.keys())
                raise ValueError(
                    f"Unsupported codec: {codec_name}. Supported: {available_codecs}"
                )

    def _is_valid_codec(self, codec_name: str) -> bool:
        """Check if a codec name is valid."""
        return (codec_name in self.IMAGE_CODEC_CONFIGS
                or codec_name in self.RAW_DATA_CODEC_CONFIGS)

    def get_codec_for_feature(self,
                              feature_type: FeatureType,
                              feature_name: Optional[str] = None) -> str:
        """Determine the appropriate codec for a given feature type and name."""

        # Check for feature-specific codec mapping first
        if feature_name and feature_name in self.feature_codecs:
            specified_codec = self.feature_codecs[feature_name]
            logger.debug(
                f"Using feature-specific codec {specified_codec} for {feature_name}"
            )

            # Validate the codec can handle this feature type
            if self._can_codec_handle_feature(specified_codec, feature_type):
                return specified_codec
            else:
                logger.warning(
                    f"Feature-specific codec {specified_codec} cannot handle feature {feature_name} "
                    f"with type {feature_type}, falling back to auto-selection"
                )

        # Determine if this is RGB image data that can use video codecs
        data_shape = feature_type.shape
        is_rgb_image = (data_shape is not None and len(data_shape) == 3
                        and data_shape[2] == 3)

        if is_rgb_image and data_shape is not None:
            # This is RGB image data - can use video codecs
            height, width = data_shape[0], data_shape[1]

            # Check if a specific video codec was provided
            if self.video_codec and self.video_codec != "auto":
                if (self.is_image_codec(self.video_codec)
                        and data_shape is not None
                        and self.is_valid_image_shape(data_shape,
                                                      self.video_codec)):
                    logger.debug(
                        f"Using specified video codec {self.video_codec} for RGB shape {data_shape}"
                    )
                    return self.video_codec
                else:
                    logger.warning(
                        f"Specified video codec {self.video_codec} doesn't support shape {data_shape}, falling back to auto-selection"
                    )

            # Check if user specified a general codec other than auto
            if self.codec != "auto" and self.is_image_codec(self.codec):
                if data_shape is not None and self.is_valid_image_shape(
                        data_shape, self.codec):
                    logger.debug(
                        f"Using user-specified image codec {self.codec} for RGB shape {data_shape}"
                    )
                    return self.codec
                else:
                    logger.warning(
                        f"User-specified codec {self.codec} doesn't support shape {data_shape}, falling back to auto-selection"
                    )

            # Auto-selection for RGB images only
            codec_preferences = [
                "libx265",
                "libx264",
                "ffv1",
                "libaom-av1",
            ]

            for codec in codec_preferences:
                if data_shape is not None and self.is_valid_image_shape(
                        data_shape, codec):
                    logger.debug(
                        f"Selected image codec {codec} for RGB shape {data_shape}"
                    )
                    return codec

            # If no image codec works for this RGB image, fall back to rawvideo
            logger.warning(
                f"No image codec supports RGB shape {data_shape}, falling back to rawvideo"
            )
            return "rawvideo"

        else:
            # This is non-RGB data (scalars, grayscale, depth, vectors, etc.) - use raw data codecs
            logger.debug(
                f"Processing non-RGB data with shape {data_shape} - using raw codec"
            )

            # Check if a specific raw codec was provided
            if self.raw_codec and self.raw_codec != "auto":
                if self.is_raw_data_codec(self.raw_codec):
                    logger.debug(
                        f"Using specified raw codec {self.raw_codec} for non-RGB data"
                    )
                    return self.raw_codec
                else:
                    logger.warning(
                        f"Specified raw codec {self.raw_codec} is not a valid raw codec, falling back to default"
                    )

            # Check if user specified a general raw codec
            if self.codec != "auto" and self.is_raw_data_codec(self.codec):
                logger.debug(
                    f"Using user-specified raw codec {self.codec} for non-RGB data"
                )
                return self.codec

            # Default to basic rawvideo for non-RGB data
            return "rawvideo"

    def _can_codec_handle_feature(self, codec: str,
                                  feature_type: FeatureType) -> bool:
        """Check if a codec can handle a specific feature type."""
        if self.is_raw_data_codec(codec):
            # Raw data codecs can handle any data type
            return True

        # Image codecs can only handle RGB images
        if self.is_image_codec(codec):
            data_shape = feature_type.shape
            if data_shape is not None and len(
                    data_shape) == 3 and data_shape[2] == 3:
                return self.is_valid_image_shape(data_shape, codec)

        return False

    def get_container_codec(self, codec: str) -> str:
        """Get the container codec name for a given codec."""
        if codec in self.IMAGE_CODEC_CONFIGS:
            return cast(str,
                        self.IMAGE_CODEC_CONFIGS[codec]["container_codec"])
        elif codec in self.RAW_DATA_CODEC_CONFIGS:
            return cast(str,
                        self.RAW_DATA_CODEC_CONFIGS[codec]["container_codec"])
        else:
            raise ValueError(f"Unknown codec {codec}")

    def get_internal_codec(self, codec: str) -> Optional[str]:
        """Get the internal codec implementation name for raw data codecs."""
        if codec in self.RAW_DATA_CODEC_CONFIGS:
            return cast(str,
                        self.RAW_DATA_CODEC_CONFIGS[codec]["internal_codec"])
        elif codec in self.IMAGE_CODEC_CONFIGS:
            # Image codecs don't have internal codecs
            return None
        else:
            raise ValueError(f"Unknown codec {codec}")

    def get_raw_codec_name(self, codec: str) -> str:
        """Get the raw codec implementation name for a given codec (legacy compatibility)."""
        internal_codec = self.get_internal_codec(codec)
        if internal_codec is not None:
            return internal_codec

        # Fallback for backward compatibility
        legacy_configs = self.CODEC_CONFIGS
        if codec in legacy_configs:
            return cast(str, legacy_configs[codec].get("raw_codec",
                                                       "pickle_raw"))

        return "pickle_raw"

    def get_pixel_format(self, codec: str,
                         feature_type: FeatureType) -> Optional[str]:
        """Get appropriate pixel format for codec and feature type."""
        if codec in self.IMAGE_CODEC_CONFIGS:
            base_format = cast(
                Optional[str],
                self.IMAGE_CODEC_CONFIGS[codec].get("pixel_format"))

            # For FFV1, use RGB24 to avoid YUV conversion issues
            if codec == "ffv1":
                data_shape = feature_type.shape
                if data_shape is not None and len(data_shape) == 3:
                    if data_shape[2] == 3:  # RGB
                        return "rgb24"
                    elif data_shape[2] == 4:  # RGBA
                        return "rgba"
                # Fallback to rgb24 for any other FFV1 case
                return "rgb24"

            return base_format

        # Raw data codecs don't use pixel formats
        return None

    def get_codec_options(self, codec: str) -> Dict[str, Any]:
        """Get codec options, using only options relevant to the specific codec type."""
        default_options = {}

        if codec in self.IMAGE_CODEC_CONFIGS:
            # Video/image codec - only use video-specific options
            options_dict = self.IMAGE_CODEC_CONFIGS[codec].get("options", {})
            default_options = cast(Dict[str, Any], options_dict).copy()
            # Only merge video-specific custom options
            default_options.update(self.video_custom_options)
        elif codec in self.RAW_DATA_CODEC_CONFIGS:
            # Raw data codec - only use raw-specific options
            options_dict = self.RAW_DATA_CODEC_CONFIGS[codec].get(
                "options", {})
            default_options = cast(Dict[str, Any], options_dict).copy()
            # Only merge raw-specific custom options
            default_options.update(self.raw_custom_options)

        return default_options

    @classmethod
    def for_transcoding_to_internal_codec(
            cls,
            internal_codec: str,
            codec_options: Optional[Dict[str, Any]] = None) -> Any:
        """Create a CodecConfig specifically for transcoding to a particular internal codec.

        This is used during transcoding operations where we need to convert between
        different raw data codec implementations (e.g., pickle_raw -> pyarrow_batch).

        Args:
            internal_codec: The target internal codec (e.g., "pyarrow_batch", "pickle_raw")
            codec_options: Options specific to the internal codec

        Returns:
            A CodecConfig instance configured for the specified internal codec
        """
        return cls._TranscodingCodecConfig(internal_codec, codec_options or {})

    class _TranscodingCodecConfig:
        """A specialized codec configuration for transcoding operations."""

        def __init__(self, target_internal_codec: str,
                     codec_options: Dict[str, Any]):
            self.target_internal_codec = target_internal_codec
            self.codec_options = codec_options

        def get_internal_codec(self, enc: str) -> str:
            """Return the target internal codec for any encoding."""
            return self.target_internal_codec

        def get_codec_options(self, enc: str) -> Dict[str, Any]:
            """Return the codec options for the target internal codec."""
            return self.codec_options

        def is_image_codec(self, codec_name: str) -> bool:
            """Check if a codec is an image/video codec."""
            return codec_name in {"libx264", "libx265", "libaom-av1", "ffv1"}

        def is_raw_data_codec(self, codec_name: str) -> bool:
            """Check if a codec is for raw/non-image data."""
            return codec_name.startswith(
                "rawvideo") or codec_name == "rawvideo"

        @property
        def RAW_DATA_CODEC_CONFIGS(self) -> Dict[str, Dict[str, Any]]:
            """Return raw data codec configurations for the target internal codec."""
            return {
                "transcoding_target": {
                    "internal_codec": self.target_internal_codec,
                    "options": self.codec_options,
                }
            }
