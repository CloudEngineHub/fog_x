"""
Codec Manager for handling codec instantiation and packet processing.

This provides an extensible way to manage codecs without case-by-case handling
in the backend implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import PacketInfo
from .codec_interface import CodecPacket, DataCodec, RawDataCodec, VideoCodec
from .codecs import (get_codec, is_raw_codec, is_video_codec,
                     list_available_codecs)

logger = logging.getLogger(__name__)


class CodecManager:
    """Manages codec instances and handles packet encoding/decoding"""

    def __init__(self):
        # Map stream_index -> codec instance
        self._stream_codecs: Dict[int, DataCodec] = {}
        # Map stream_index -> codec configuration
        self._stream_configs: Dict[int, Dict[str, Any]] = {}

    def create_codec_for_stream(
        self,
        stream_index: int,
        container_encoding: str,
        codec_config: Any,
        feature_type: Any = None,
        stream: Any = None,
    ) -> Optional[DataCodec]:
        """Create and configure a codec for a stream.

        Args:
            stream_index: Index of the stream
            container_encoding: The container codec (e.g., "libx264", "rawvideo")
            codec_config: Codec configuration object
            feature_type: Feature type information
            stream: Stream object (for video codecs)
        """
        # Determine the actual codec implementation to use
        codec_impl_name = self._determine_codec_implementation(
            container_encoding, codec_config)

        # Get codec configuration
        config = self._build_codec_config(codec_impl_name, codec_config,
                                          feature_type, container_encoding)

        # Create codec instance
        codec = self._create_codec_instance(codec_impl_name, config)

        # Configure the codec if needed
        if isinstance(codec, VideoCodec) and stream is not None:
            codec.configure_stream(stream, feature_type)

        # Cache the codec and its config
        self._stream_codecs[stream_index] = codec
        self._stream_configs[stream_index] = config

        logger.debug(
            f"Created codec {codec_impl_name} for stream {stream_index} (container: {container_encoding})"
        )
        return codec

    def _determine_codec_implementation(self, container_encoding: str,
                                        codec_config: Any) -> str:
        """Determine the actual codec implementation to use.

        Args:
            container_encoding: The container codec (e.g., "libx264", "rawvideo")
            codec_config: Codec configuration object

        Returns:
            The codec implementation name to use
        """
        # For image/video codecs, use the container encoding directly
        if codec_config.is_image_codec(container_encoding):
            return container_encoding

        # For raw data, determine the internal codec implementation
        elif container_encoding == "rawvideo":
            # Use codec config to determine the internal implementation
            if hasattr(codec_config, "get_internal_codec"):
                # For transcoding cases, we might have a specialized config that knows
                # exactly which internal codec to use
                internal_codec = codec_config.get_internal_codec("rawvideo")
                if internal_codec:
                    return internal_codec
                else:
                    return "pickle_raw"
            else:
                return "pickle_raw"

        else:
            raise ValueError(
                f"Unknown container encoding: {container_encoding}")

    def _create_codec_instance(self, codec_impl_name: str,
                               config: Dict[str, Any]) -> DataCodec:
        """Create a codec instance with the given configuration."""
        try:
            # get_codec passes codec_impl_name as the first argument to the codec class
            # For PyAVVideoCodec, it can handle codec_name being passed either as:
            # 1. First positional arg: PyAVVideoCodec('libx264', ...)
            # 2. Keyword arg: PyAVVideoCodec(codec_name='libx264', ...)
            # Since get_codec doesn't pass codec_name to the constructor, we need to add it
            # get_codec takes codec_name as its first positional parameter
            # So we must not include 'codec_name' in the kwargs to avoid duplicate argument error
            config_without_codec_name = {
                k: v
                for k, v in config.items() if k != "codec_name"
            }

            if is_video_codec(codec_impl_name):
                # PyAVVideoCodec needs codec_name in its constructor kwargs
                # Since get_codec doesn't pass the codec_name to the constructor,
                # we need to add it back to the config
                config_without_codec_name["codec_name"] = codec_impl_name

            codec = get_codec(codec_impl_name, **config_without_codec_name)

            return codec
        except Exception as e:
            logger.error(f"Failed to create codec {codec_impl_name}: {e}")
            raise

    def get_codec_for_stream(self, stream_index: int) -> Optional[DataCodec]:
        """Get the codec instance for a stream"""
        return self._stream_codecs.get(stream_index)

    def encode_data(self,
                    stream_index: int,
                    data: Any,
                    timestamp: int,
                    stream: Any = None) -> List[PacketInfo]:
        """Encode data using the appropriate codec for the stream"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            logger.error(f"No codec found for stream {stream_index}")
            return []

        try:
            # Encode data to codec packets
            codec_packets = codec.encode(data, timestamp)

            # Convert to PacketInfo objects
            packet_infos = []
            for codec_packet in codec_packets:
                packet_info = self._codec_packet_to_packet_info(
                    codec_packet, stream_index, timestamp, stream)
                packet_infos.append(packet_info)

            return packet_infos

        except Exception as e:
            logger.error(
                f"Failed to encode data for stream {stream_index}: {e}")
            return []

    def flush_stream(self,
                     stream_index: int,
                     stream: Any = None) -> List[PacketInfo]:
        """Flush any buffered data from a stream's codec"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            return []

        try:
            codec_packets = codec.flush()
            packet_infos = []

            for codec_packet in codec_packets:
                packet_info = self._codec_packet_to_packet_info(
                    codec_packet, stream_index, None, stream)
                packet_infos.append(packet_info)

            return packet_infos

        except Exception as e:
            logger.error(f"Failed to flush stream {stream_index}: {e}")
            return []

    def decode_packet(self, packet_info: PacketInfo) -> Any:
        """Decode a packet using the appropriate codec"""
        stream_index = packet_info.stream_index
        codec = self._stream_codecs.get(stream_index)

        if codec is None:
            logger.warning(
                f"No codec found for stream {stream_index}, using fallback")
            return self._fallback_decode(packet_info)

        try:
            # Convert PacketInfo to CodecPacket
            codec_packet = self._packet_info_to_codec_packet(
                packet_info, codec)

            # Decode using codec
            return codec.decode(codec_packet)

        except Exception as e:
            logger.error(
                f"Failed to decode packet for stream {stream_index}: {e}")
            return self._fallback_decode(packet_info)

    def clear_stream_codecs(self):
        """Clear all stream codecs"""
        self._stream_codecs.clear()
        self._stream_configs.clear()

    def get_codec_info(self, stream_index: int) -> Optional[Dict[str, Any]]:
        """Get information about the codec for a stream"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            return None

        return {
            "codec_name": codec.get_codec_name(),
            "supports_seeking": codec.supports_seeking(),
            "is_video_codec": isinstance(codec, VideoCodec),
            "is_raw_codec": isinstance(codec, RawDataCodec),
            "config": self._stream_configs.get(stream_index, {}),
        }

    # Private helper methods

    def _build_codec_config(
        self,
        codec_impl_name: str,
        codec_config: Any,
        feature_type: Any,
        container_encoding: str,
    ) -> Dict[str, Any]:
        """Build configuration dictionary for codec creation"""
        config = {}

        # Add codec name for video codecs that need it
        if is_video_codec(codec_impl_name):
            # For video codecs, pass codec_name as first positional argument
            # and other config as keyword arguments
            if hasattr(codec_config, "get_pixel_format"):
                pixel_fmt = codec_config.get_pixel_format(
                    container_encoding, feature_type)
                if pixel_fmt:
                    config["pixel_format"] = pixel_fmt

            if hasattr(codec_config, "get_codec_options"):
                codec_opts = codec_config.get_codec_options(container_encoding)
                if codec_opts:
                    config["options"] = codec_opts

        elif is_raw_codec(codec_impl_name):
            # Add raw codec specific config, but filter based on actual codec implementation
            if hasattr(codec_config, "get_codec_options"):
                # For raw codecs, we need to determine which rawvideo variant was requested
                # Since we might not have that info directly, we'll try to get options from
                # the internal codec configuration
                raw_codec_options = {}

                # Try to get options from the raw data codec configs
                if hasattr(codec_config, "RAW_DATA_CODEC_CONFIGS"):
                    for (
                            raw_codec_name,
                            raw_config,
                    ) in codec_config.RAW_DATA_CODEC_CONFIGS.items():
                        if raw_config.get("internal_codec") == codec_impl_name:
                            raw_codec_options = raw_config.get("options", {})
                            break

                # Merge with any custom options
                if raw_codec_options:
                    filtered_opts = self._filter_codec_options(
                        codec_impl_name, raw_codec_options)
                    config.update(filtered_opts)

        return config

    def _filter_codec_options(self, codec_name: str,
                              codec_options: Dict[str, Any]) -> Dict[str, Any]:
        """Filter codec options based on what the specific codec implementation can handle"""
        if codec_name == "pickle_raw":
            # PickleRawCodec doesn't accept any constructor parameters
            return {}
        elif codec_name == "pyarrow_batch":
            # PyArrowBatchCodec accepts batch_size and compression
            allowed_options = {"batch_size", "compression"}
            return {
                k: v
                for k, v in codec_options.items() if k in allowed_options
            }
        else:
            # For unknown raw codecs, pass all options (backward compatibility)
            return codec_options

    def _codec_packet_to_packet_info(
        self,
        codec_packet: CodecPacket,
        stream_index: int,
        default_timestamp: Optional[int],
        stream: Any = None,
    ) -> PacketInfo:
        """Convert a CodecPacket to PacketInfo"""
        # Get time base from stream if available
        if stream is not None and hasattr(stream, "time_base"):
            time_base = (stream.time_base.numerator,
                         stream.time_base.denominator)
        else:
            time_base = (1, 1000)  # Default millisecond time base

        return PacketInfo(
            data=codec_packet.data,
            pts=codec_packet.metadata.get("pts", default_timestamp),
            dts=codec_packet.metadata.get("dts", default_timestamp),
            stream_index=stream_index,
            time_base=time_base,
            is_keyframe=codec_packet.metadata.get("is_keyframe",
                                                  codec_packet.seekable),
        )

    def _packet_info_to_codec_packet(self, packet_info: PacketInfo,
                                     codec: DataCodec) -> CodecPacket:
        """Convert PacketInfo to CodecPacket for decoding"""
        return CodecPacket(
            data=packet_info.data,
            metadata={
                "pts": packet_info.pts,
                "dts": packet_info.dts,
                "codec": codec.get_codec_name(),
                "time_base": packet_info.time_base,
            },
            seekable=packet_info.is_keyframe,
        )

    def _fallback_decode(self, packet_info: PacketInfo) -> Any:
        """Fallback decoding using pickle"""
        try:
            import pickle

            return pickle.loads(packet_info.data)
        except Exception as e:
            logger.error(f"Fallback decode failed: {e}")
            return packet_info.data
