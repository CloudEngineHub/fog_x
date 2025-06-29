from __future__ import annotations

"""PyAV-backed implementation of the ContainerBackend interface.

This module converts the abstract operations defined in
:`robodm.backend.base.ContainerBackend` into concrete calls against the
`PyAV` API so that the rest of the codebase can remain backend-agnostic.

The guiding principle is **minimum interference** with the existing logic in
`robodm.trajectory.Trajectory`: wherever that class already manipulates
`PyAV` primitives directly, this backend returns or accepts those same
objects so we do **not** have to rewrite the fragile frame-handling code.
"""

import logging
import os
import pickle
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import av
import numpy as np

from robodm import FeatureType
from robodm.backend.codec_config import CodecConfig

from .base import ContainerBackend, PacketInfo, StreamConfig, StreamMetadata
from .codec_manager import CodecManager

logger = logging.getLogger(__name__)


class PyAVBackend(ContainerBackend):
    """ContainerBackend implementation that relies on the PyAV library.

    Notes
    -----
    * The backend keeps a reference to the underlying :class:`av.container.InputContainer`
      or :class:`av.container.OutputContainer` in ``self.container`` so that legacy
      code can keep using `container.mux(...)`, `container.streams`, etc.
    * All timestamps are interpreted in **milliseconds** – this mirrors the rest
      of the codebase where the time-base is hard-coded to ``Fraction(1, 1000)``.
    """

    DEFAULT_FORMAT: str = "matroska"

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def __init__(self, container_format: str | None = None) -> None:
        self.container_format: str = container_format or self.DEFAULT_FORMAT
        self.container: av.container.Container | None = None
        # Map index -> av.Stream for quick lookup
        self._idx_to_stream: Dict[int, av.stream.Stream] = {}
        # Codec manager for handling encoding/decoding
        self.codec_manager = CodecManager()

    # ------------------------------------------------------------------
    # API implementation
    # ------------------------------------------------------------------
    def open(self, path: str,
             mode: str) -> None:  # noqa: D401  (docstring inherited)
        if mode not in {"r", "w"}:
            raise ValueError("mode must be 'r' or 'w'")
        self.container = av.open(
            path, mode=mode,
            format=self.container_format)  # type: ignore[call-overload]
        # Populate mapping for existing streams (in read mode).
        if mode == "r":
            self._idx_to_stream = {
                s.index: s
                for s in self.container.streams  # type: ignore[index]
            }

    def close(self) -> None:
        if self.container is not None:
            self.container.close()  # type: ignore[attr-defined]
            self.container = None
            self._idx_to_stream.clear()
            self.codec_manager.clear_stream_codecs()

    def get_streams(self) -> List[StreamMetadata]:
        out: List[StreamMetadata] = []
        for idx, stream in self._idx_to_stream.items():
            fn = stream.metadata.get("FEATURE_NAME", f"stream_{idx}")
            ft = stream.metadata.get("FEATURE_TYPE", "unknown")
            enc = stream.codec_context.codec.name
            tb = ((stream.time_base.numerator, stream.time_base.denominator)
                  if stream.time_base is not None else (1, 1000))
            out.append(
                StreamMetadata(
                    feature_name=fn,
                    feature_type=ft,
                    encoding=enc,
                    time_base=tb,
                ))
        return out

    # ------------------------------------------------------------------
    # New containerization abstractions
    # ------------------------------------------------------------------

    def encode_data_to_packets(
        self,
        data: Any,
        stream_index: int,
        timestamp: int,
        codec_config: Any,
        force_direct_encoding: bool = False,
    ) -> List[PacketInfo]:
        """Encode arbitrary data into packets with timestamp handling

        Args:
            data: Data to encode
            stream_index: Target stream index
            timestamp: Timestamp in milliseconds
            codec_config: Codec configuration
            force_direct_encoding: If True, encode directly to target format instead of rawvideo
        """
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        container_encoding = stream.codec_context.codec.name

        # If force_direct_encoding is True, bypass rawvideo intermediate step
        if force_direct_encoding and container_encoding != "rawvideo":
            return self._encode_directly_to_target(data, stream_index,
                                                   timestamp, codec_config)

        # Create codec if it doesn't exist
        codec = self.codec_manager.get_codec_for_stream(stream_index)
        if codec is None:
            feature_type = self._get_feature_type_from_stream(stream)
            codec = self.codec_manager.create_codec_for_stream(
                stream_index, container_encoding, codec_config, feature_type,
                stream)

        # Use codec manager to encode data
        if codec is not None:
            packets = self.codec_manager.encode_data(stream_index, data,
                                                     timestamp, stream)
            if packets:
                return packets

        return []

    def _encode_directly_to_target(self, data: Any, stream_index: int,
                                   timestamp: int,
                                   codec_config: Any) -> List[PacketInfo]:
        """Encode data directly to the target codec format without intermediate rawvideo step"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        container_encoding = stream.codec_context.codec.name

        if container_encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            # Direct video encoding
            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                frame = self._create_frame(data, stream)
                frame.time_base = stream.time_base
                frame.pts = timestamp
                frame.dts = timestamp

                packets = []
                for pkt in stream.encode(frame):  # type: ignore[attr-defined]
                    packets.append(
                        PacketInfo(
                            data=bytes(pkt),
                            pts=pkt.pts,
                            dts=pkt.dts,
                            stream_index=stream_index,
                            time_base=(
                                (stream.time_base.numerator
                                 if stream.time_base is not None else 1),
                                (stream.time_base.denominator
                                 if stream.time_base is not None else 1000),
                            ),
                            is_keyframe=(bool(pkt.is_keyframe) if hasattr(
                                pkt, "is_keyframe") else False),
                        ))
                return packets

        # Fallback to legacy encoding if direct encoding isn't supported
        return self._legacy_encode_fallback(data, stream_index, timestamp,
                                            stream)

    def _get_feature_type_from_stream(self, stream: Any) -> Any:
        """Extract feature type information from stream metadata"""
        # This is a placeholder - in practice you might parse the FEATURE_TYPE metadata
        # or use other mechanisms to get the actual FeatureType object
        return None

    def _legacy_encode_fallback(self, data: Any, stream_index: int,
                                timestamp: int,
                                stream: Any) -> List[PacketInfo]:
        """Legacy encoding fallback"""
        encoding = stream.codec_context.codec.name

        if (encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}
                and isinstance(data, np.ndarray) and len(data.shape) >= 2):
            # Legacy video encoding
            frame = self._create_frame(data, stream)
            frame.time_base = stream.time_base
            frame.pts = timestamp
            frame.dts = timestamp

            packets = []
            for pkt in stream.encode(frame):  # type: ignore[attr-defined]
                packets.append(
                    PacketInfo(
                        data=bytes(pkt),
                        pts=pkt.pts,
                        dts=pkt.dts,
                        stream_index=stream_index,
                        time_base=(
                            (stream.time_base.numerator
                             if stream.time_base is not None else 1),
                            (stream.time_base.denominator
                             if stream.time_base is not None else 1000),
                        ),
                        is_keyframe=(bool(pkt.is_keyframe) if hasattr(
                            pkt, "is_keyframe") else False),
                    ))
            return packets
        else:
            # Legacy pickle encoding
            if isinstance(data, np.ndarray):
                payload = pickle.dumps(data)
            else:
                payload = pickle.dumps(data)

            return [
                PacketInfo(
                    data=payload,
                    pts=timestamp,
                    dts=timestamp,
                    stream_index=stream_index,
                    time_base=(
                        (stream.time_base.numerator
                         if stream.time_base is not None else 1),
                        (stream.time_base.denominator
                         if stream.time_base is not None else 1000),
                    ),
                    is_keyframe=True,
                )
            ]

    def flush_all_streams(self) -> List[PacketInfo]:
        """Flush all streams and return all buffered packets"""
        packets: List[PacketInfo] = []
        for stream_index in self._idx_to_stream:
            packets.extend(self._flush_stream(stream_index))
        return packets

    def _flush_stream(self, stream_index: int) -> List[PacketInfo]:
        """Internal helper to flush a single stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]

        # Try codec manager first
        packets = self.codec_manager.flush_stream(stream_index, stream)
        if packets:
            return packets

        # Fallback to legacy PyAV stream flushing for video codecs
        packets = []
        try:
            # Flush the encoder
            for pkt in stream.encode(None):  # type: ignore[attr-defined]
                packets.append(
                    PacketInfo(
                        data=bytes(pkt),
                        pts=pkt.pts,
                        dts=pkt.dts,
                        stream_index=stream_index,
                        time_base=(
                            (stream.time_base.numerator
                             if stream.time_base is not None else 1),
                            (stream.time_base.denominator
                             if stream.time_base is not None else 1000),
                        ),
                        is_keyframe=(bool(pkt.is_keyframe) if hasattr(
                            pkt, "is_keyframe") else False),
                    ))
        except av.error.EOFError:
            # Expected when encoder is fully flushed
            pass
        except Exception as e:
            logger.error(f"Error flushing stream {stream_index}: {e}")

        return packets

    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """Mux a PacketInfo object to the container"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        if packet_info.stream_index not in self._idx_to_stream:
            raise ValueError(
                f"No stream with index {packet_info.stream_index}")

        pkt = av.Packet(packet_info.data)
        pkt.pts = packet_info.pts
        pkt.dts = packet_info.dts
        pkt.time_base = Fraction(*packet_info.time_base)
        pkt.stream = self._idx_to_stream[packet_info.stream_index]

        self.container.mux(pkt)  # type: ignore[attr-defined]

    def transcode_container(
        self,
        input_path: str,
        output_path: str,
        stream_configs: Dict[int, StreamConfig],
        visualization_feature: Optional[str] = None,
    ) -> None:
        """Transcode a container from one format/encoding to another"""

        # Open input container
        input_container = av.open(input_path,
                                  mode="r",
                                  format=self.container_format)
        input_streams = list(input_container.streams)

        # Create output container
        output_container = av.open(output_path,
                                   mode="w",
                                   format=self.container_format)

        # Sort streams to prioritize visualization feature
        def get_stream_priority(stream):
            feature_name = stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                return (3, stream.index)

            # Highest priority: specified visualization_feature
            if visualization_feature and feature_name == visualization_feature:
                return (0, stream.index)

            # Second priority: streams that will become video-encoded
            if stream.index in stream_configs:
                config = stream_configs[stream.index]
                if config.encoding != "rawvideo":
                    return (1, stream.index)

            # Third priority: everything else
            return (2, stream.index)

        sorted_streams = sorted(input_streams, key=get_stream_priority)

        # Create output streams
        stream_mapping: Dict[int, int] = {}
        for input_stream in sorted_streams:
            feature_name = input_stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                continue

            if input_stream.index in stream_configs:
                config = stream_configs[input_stream.index]
                output_stream_idx = self._create_output_stream(
                    output_container, config)
            else:
                # Copy existing stream configuration
                config = StreamConfig(
                    feature_name=feature_name,
                    feature_type=input_stream.metadata.get(
                        "FEATURE_TYPE", "unknown"),
                    encoding=input_stream.codec_context.codec.name,
                )
                output_stream_idx = self._create_output_stream(
                    output_container, config)

            stream_mapping[input_stream.index] = output_stream_idx

        # Process packets
        packets_muxed = 0
        for packet in input_container.demux(input_streams):
            if not self.validate_packet(packet):
                logger.debug(f"Skipping invalid packet: {packet}")
                continue

            if packet.stream.index not in stream_mapping:
                continue

            output_stream_idx = stream_mapping[packet.stream.index]
            output_stream = output_container.streams[output_stream_idx]

            # Get transcoding configuration
            original_container_codec = packet.stream.codec_context.codec.name
            original_selected_codec = packet.stream.metadata.get(
                "SELECTED_CODEC", original_container_codec)

            target_config = stream_configs.get(packet.stream.index)

            if target_config:
                target_container_codec = target_config.encoding
                target_selected_codec = getattr(target_config,
                                                "selected_codec",
                                                target_config.encoding)

                # Determine transcoding strategy
                needs_transcoding = self._needs_transcoding(
                    original_container_codec,
                    original_selected_codec,
                    target_container_codec,
                    target_selected_codec,
                    packet.stream.metadata,
                    target_config,
                )

                if needs_transcoding:
                    success = self._transcode_packet(
                        packet,
                        output_stream,
                        output_container,
                        original_container_codec,
                        target_container_codec,
                        original_selected_codec,
                        target_selected_codec,
                        target_config,
                    )
                    if success:
                        packets_muxed += 1
                else:
                    # Direct remux
                    packet.stream = output_stream
                    output_container.mux(packet)
                    packets_muxed += 1
            else:
                # No target config, direct remux
                packet.stream = output_stream
                output_container.mux(packet)
                packets_muxed += 1

        # Flush all output streams
        for stream in output_container.streams:
            try:
                for packet in stream.encode(
                        None):  # type: ignore[attr-defined]
                    output_container.mux(packet)
                    packets_muxed += 1
            except Exception as e:
                logger.error(f"Error flushing output stream {stream}: {e}")

        logger.debug(f"Transcoding complete: {packets_muxed} packets muxed")

        input_container.close()  # type: ignore[attr-defined]
        output_container.close()  # type: ignore[attr-defined]

    def create_container_with_new_streams(
        self,
        original_path: str,
        new_path: str,
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig],
    ) -> Dict[int, int]:
        """Create a new container with existing streams plus new ones"""

        # Open original container
        original_container = av.open(original_path,
                                     mode="r",
                                     format=self.container_format)
        original_stream_objects = list(original_container.streams)

        # Create new container
        new_container = av.open(new_path,
                                mode="w",
                                format=self.container_format)

        stream_mapping: Dict[int, int] = {}

        # Add existing streams
        for old_idx, config in existing_streams:
            new_idx = self._create_output_stream(new_container, config)
            stream_mapping[old_idx] = new_idx

        # Add new streams
        for config in new_stream_configs:
            new_idx = self._create_output_stream(new_container, config)
            # New streams don't have an old index to map from

        # Copy existing packets
        for packet in original_container.demux(original_stream_objects):
            if not self.validate_packet(packet):
                continue

            if packet.stream.index in stream_mapping:
                new_stream_idx = stream_mapping[packet.stream.index]
                packet.stream = new_container.streams[new_stream_idx]
                new_container.mux(packet)

        original_container.close()  # type: ignore[attr-defined]

        # Keep new container open and update our state
        if self.container is not None:
            self.container.close()  # type: ignore[attr-defined]
        self.container = new_container
        self._idx_to_stream = {s.index: s for s in new_container.streams}

        return stream_mapping

    def validate_packet(self, packet: Any) -> bool:
        """Check if a packet has valid pts/dts"""
        # Only check pts like the original code - some packets may not have dts
        return packet.pts is not None

    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Get an iterator for demuxing specific streams"""
        if self.container is None:
            raise RuntimeError("Container not opened")

        # Get the actual stream objects for the given indices
        streams = [
            self._idx_to_stream[idx] for idx in stream_indices
            if idx in self._idx_to_stream
        ]
        return self.container.demux(streams)  # type: ignore[attr-defined]

    def seek_container(self,
                       timestamp: int,
                       stream_index: int,
                       any_frame: bool = True) -> None:
        """Seek the container to a specific timestamp"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        self.container.seek(timestamp, stream=stream,
                            any_frame=any_frame)  # type: ignore[attr-defined]

    def decode_stream_frames(self,
                             stream_index: int,
                             packet_data: Optional[bytes] = None) -> List[Any]:
        """Decode frames from a stream, optionally with packet data"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]

        if packet_data is None:
            # Flush decoder
            return list(stream.decode(None))  # type: ignore[attr-defined]
        else:
            # Decode specific packet
            pkt = av.Packet(packet_data)
            pkt.stream = stream
            return list(pkt.decode())

    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get the codec name for a stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        return stream.codec_context.codec.name

    def convert_frame_to_array(self,
                               frame: Any,
                               feature_type: Any,
                               format: str = "rgb24") -> Any:
        """Convert a backend-specific frame to numpy array"""
        import pickle

        # Try to use codec manager for decoding if frame is a PacketInfo
        if hasattr(frame, "stream_index") and hasattr(frame, "data"):
            try:
                return self.codec_manager.decode_packet(frame)
            except Exception as e:
                logger.warning(f"Codec manager decode failed: {e}")

        # Handle pickled data (rawvideo packets) - legacy support
        if isinstance(frame, bytes):
            return pickle.loads(frame)

        # Handle PyAV video frames
        if hasattr(frame, "to_ndarray"):
            # Check if this is RGB data that should be decoded as RGB24
            if (hasattr(feature_type, "shape") and feature_type.shape
                    and len(feature_type.shape) == 3
                    and feature_type.shape[2] == 3):
                arr = frame.to_ndarray(format=format)
            else:
                # For non-RGB data, this might be an issue but handle gracefully
                arr = frame.to_ndarray(format=format)

            # Reshape if needed
            if hasattr(feature_type, "shape") and feature_type.shape:
                arr = arr.reshape(feature_type.shape)

            return arr

        # Fallback - return as is
        return frame

    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if a stream exists for a given feature name"""
        for stream_idx, stream in self._idx_to_stream.items():
            if stream.metadata.get("FEATURE_NAME") == feature_name:
                return stream_idx
        return None

    # ------------------------------------------------------------------
    # High-level helpers that map directly from Trajectory logic
    # ------------------------------------------------------------------
    def add_stream_for_feature(
        self,
        feature_name: str,
        feature_type: "FeatureType",
        codec_config: "CodecConfig",
        encoding: str | None = None,
    ) -> "av.stream.Stream":
        """Create a new stream inside the currently opened container.

        This mirrors the logic previously found in
        ``Trajectory._add_stream_to_container`` so that that method can now be
        reduced to a thin wrapper that delegates to this backend.
        """

        if self.container is None:
            raise RuntimeError("Container not opened")

        # Determine encoding if not explicitly provided.
        selected_codec = encoding or codec_config.get_codec_for_feature(
            feature_type, feature_name)

        # Get the appropriate container codec
        container_codec = codec_config.get_container_codec(selected_codec)

        # Create stream with container codec
        stream = self.container.add_stream(
            container_codec)  # type: ignore[attr-defined]

        # Configure stream for image codecs
        if codec_config.is_image_codec(container_codec):
            shape = feature_type.shape
            if shape is not None and len(shape) >= 2:
                # Only set width/height for video streams
                if hasattr(stream, "width") and hasattr(stream, "height"):
                    stream.width = shape[1]  # type: ignore[attr-defined]
                    stream.height = shape[0]  # type: ignore[attr-defined]

            pixel_fmt = codec_config.get_pixel_format(selected_codec,
                                                      feature_type)
            if pixel_fmt and hasattr(stream, "pix_fmt"):
                stream.pix_fmt = pixel_fmt  # type: ignore[attr-defined]

            codec_opts = codec_config.get_codec_options(selected_codec)
            if codec_opts:
                # Convert all option values to strings since PyAV expects string values
                string_options = {k: str(v) for k, v in codec_opts.items()}
                stream.codec_context.options = string_options

        # Metadata and time-base
        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.metadata[
            "SELECTED_CODEC"] = selected_codec  # Store the selected codec

        # For raw data codecs, store the internal codec implementation
        if codec_config.is_raw_data_codec(selected_codec):
            internal_codec = codec_config.get_internal_codec(selected_codec)
            if internal_codec:
                stream.metadata["INTERNAL_CODEC"] = internal_codec

        stream.time_base = Fraction(1, 1000)

        self._idx_to_stream[stream.index] = stream
        return stream

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_output_stream(self, container: av.container.OutputContainer,
                              config: StreamConfig) -> int:
        """Helper to create a stream in an output container"""
        # Use the encoding directly as the container codec (it should already be the container codec)
        stream = container.add_stream(config.encoding)

        # Configure image codec settings
        if config.encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            if (config.width and config.height and hasattr(stream, "width")
                    and hasattr(stream, "height")):
                stream.width = config.width  # type: ignore[attr-defined]
                stream.height = config.height  # type: ignore[attr-defined]
            elif (hasattr(config.feature_type, "shape")
                  and hasattr(stream, "width") and hasattr(stream, "height")):
                shape = getattr(config.feature_type, "shape", None)
                if shape and len(shape) >= 2:
                    stream.width = shape[1]  # type: ignore[attr-defined]
                    stream.height = shape[0]  # type: ignore[attr-defined]

            if config.pixel_format and hasattr(stream, "pix_fmt"):
                stream.pix_fmt = config.pixel_format  # type: ignore[attr-defined]

            if config.codec_options:
                # Convert all option values to strings since PyAV expects string values
                string_options = {
                    k: str(v)
                    for k, v in config.codec_options.items()
                }
                stream.codec_context.options = string_options

        # Set metadata
        stream.metadata["FEATURE_NAME"] = config.feature_name
        stream.metadata["FEATURE_TYPE"] = str(config.feature_type)
        stream.metadata[
            "SELECTED_CODEC"] = config.encoding  # Use consistent naming

        # Store internal codec information for rawvideo streams
        if config.encoding == "rawvideo" and config.internal_codec:
            stream.metadata["INTERNAL_CODEC"] = config.internal_codec

        stream.time_base = Fraction(1, 1000)

        return stream.index

    # The following helpers replicate the fragile image handling logic that
    # previously lived in Trajectory.

    def _create_frame(self, image_array, stream):
        import numpy as _np

        image_array = _np.array(image_array)
        encoding = stream.codec_context.codec.name

        # Convert to uint8 if needed
        if image_array.dtype == _np.float32:
            image_array = _np.clip(image_array * 255, 0, 255).astype(_np.uint8)
        elif image_array.dtype != _np.uint8:
            if _np.issubdtype(image_array.dtype, _np.integer):
                image_array = _np.clip(image_array, 0, 255).astype(_np.uint8)
            else:
                image_array = _np.clip(image_array * 255, 0,
                                       255).astype(_np.uint8)

        # Only handle RGB images (HxWx3)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(
                "Video codecs only support RGB images with shape (H, W, 3). "
                f"Got shape {image_array.shape}.")

        # Create RGB frame
        frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")

        # Get the configured pixel format for this stream
        configured_pix_fmt = stream.pix_fmt

        # Convert to the configured pixel format if different from RGB24
        if configured_pix_fmt and configured_pix_fmt != "rgb24":
            frame = frame.reformat(format=configured_pix_fmt)

        return frame

    def _needs_transcoding(
        self,
        original_container_codec: str,
        original_selected_codec: str,
        target_container_codec: str,
        target_selected_codec: str,
        original_metadata: Dict[str, Any],
        target_config: Any,
    ) -> bool:
        """Determine if transcoding is needed between codecs."""

        # If container codecs are different, we need transcoding
        if original_container_codec != target_container_codec:
            return True

        # If both use rawvideo container, check internal codec differences
        if (original_container_codec == "rawvideo"
                and target_container_codec == "rawvideo"):
            original_internal = original_metadata.get("INTERNAL_CODEC",
                                                      "pickle_raw")
            target_internal = getattr(target_config, "internal_codec", None)

            # Need transcoding if internal codecs differ
            if target_internal and original_internal != target_internal:
                return True

        return False

    def _transcode_packet(
        self,
        packet: Any,
        output_stream: Any,
        output_container: Any,
        original_container_codec: str,
        target_container_codec: str,
        original_selected_codec: str,
        target_selected_codec: str,
        target_config: Any,
    ) -> bool:
        """Transcode a packet between different codecs."""

        try:
            # Handle rawvideo -> image codec transcoding
            if original_container_codec == "rawvideo" and target_container_codec in {
                    "libx264",
                    "libx265",
                    "libaom-av1",
                    "ffv1",
            }:
                return self._transcode_raw_to_image(packet, output_stream,
                                                    output_container,
                                                    target_config)

            # Handle image codec -> rawvideo transcoding
            elif (original_container_codec
                  in {"libx264", "libx265", "libaom-av1", "ffv1"}
                  and target_container_codec == "rawvideo"):
                return self._transcode_image_to_raw(packet, output_stream,
                                                    output_container,
                                                    target_config)

            # Handle image codec -> image codec transcoding
            elif original_container_codec in {
                    "libx264",
                    "libx265",
                    "libaom-av1",
                    "ffv1",
            } and target_container_codec in {
                    "libx264",
                    "libx265",
                    "libaom-av1",
                    "ffv1",
            }:
                return self._transcode_image_to_image(packet, output_stream,
                                                      output_container,
                                                      target_config)

            # Handle rawvideo internal codec transcoding
            elif (original_container_codec == "rawvideo"
                  and target_container_codec == "rawvideo"):
                return self._transcode_raw_internal(packet, output_stream,
                                                    output_container,
                                                    target_config)

            else:
                logger.warning(
                    f"Unsupported transcoding: {original_container_codec} -> {target_container_codec}"
                )
                return False

        except Exception as e:
            logger.error(f"Transcoding failed: {e}")
            return False

    def _transcode_raw_to_image(self, packet: Any, output_stream: Any,
                                output_container: Any,
                                target_config: Any) -> bool:
        """Transcode from rawvideo to image codec."""
        # Decode rawvideo packet (usually pickled data)
        data = pickle.loads(bytes(packet))

        # Create image frame
        frame = self._create_frame(data, output_stream)
        frame.time_base = output_stream.time_base
        frame.pts = packet.pts
        frame.dts = packet.dts

        # Encode and mux
        for new_packet in output_stream.encode(
                frame):  # type: ignore[attr-defined]
            new_packet.stream = output_stream
            output_container.mux(new_packet)

        return True

    def _transcode_image_to_raw(self, packet: Any, output_stream: Any,
                                output_container: Any,
                                target_config: Any) -> bool:
        """Transcode from image codec to rawvideo."""
        # This would require decoding the image packet first
        # For now, we'll log this as unsupported
        logger.warning("Image to raw transcoding not yet implemented")
        return False

    def _transcode_image_to_image(self, packet: Any, output_stream: Any,
                                  output_container: Any,
                                  target_config: Any) -> bool:
        """Transcode between different image codecs."""
        # This would require decoding and re-encoding
        # For now, we'll log this as unsupported
        logger.warning("Image to image transcoding not yet implemented")
        return False

    def _transcode_raw_internal(self, packet: Any, output_stream: Any,
                                output_container: Any,
                                target_config: Any) -> bool:
        """Transcode between different rawvideo internal codecs."""
        try:
            # Create a temporary codec manager for transcoding
            transcode_codec_manager = CodecManager()

            target_internal_codec = getattr(target_config, "internal_codec",
                                            None)
            if not target_internal_codec:
                return False

            # Create transcoding-specific codec config
            from robodm.backend.codec_config import CodecConfig

            transcoding_codec_config = CodecConfig.for_transcoding_to_internal_codec(
                target_internal_codec, target_config.codec_options or {})

            # Create codec for the target internal encoding
            codec = transcode_codec_manager.create_codec_for_stream(
                output_stream.index,
                "rawvideo",  # Container codec is always rawvideo
                transcoding_codec_config,
                target_config.feature_type,
                output_stream,
            )

            if codec:
                # Decode original data using pickle (legacy format)
                original_data = pickle.loads(bytes(packet))

                # Encode using the new codec
                codec_packets = codec.encode(original_data, packet.pts)

                # Convert codec packets to PyAV packets and mux
                for codec_packet in codec_packets:
                    new_packet = av.Packet(codec_packet.data)
                    new_packet.pts = codec_packet.metadata.get(
                        "pts", packet.pts)
                    new_packet.dts = codec_packet.metadata.get(
                        "dts", packet.pts)
                    new_packet.time_base = output_stream.time_base
                    new_packet.stream = output_stream

                    output_container.mux(new_packet)

                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to transcode internal codec: {e}")
            return False

    def create_streams_for_batch_data(
        self,
        sample_data: Dict[str, Any],
        codec_config: Any,
        feature_name_separator: str = "/",
        visualization_feature: Optional[str] = None,
    ) -> Dict[str, int]:
        """Create optimized streams for batch data processing.

        Analyzes sample data to determine optimal codec for each feature
        and creates streams with target codec directly. Respects visualization_feature
        ordering to prioritize visualization streams first.

        Args:
            sample_data: Sample data dict to analyze feature types
            codec_config: Codec configuration
            feature_name_separator: Separator for nested feature names
            visualization_feature: Optional feature name to prioritize as first stream for visualization

        Returns:
            Dict mapping feature names to stream indices
        """
        if self.container is None:
            raise RuntimeError("Container not opened")

        from robodm import FeatureType
        from robodm.utils.flatten import _flatten_dict

        # Flatten the sample data
        flattened_data = _flatten_dict(sample_data, sep=feature_name_separator)

        # Sort features to prioritize visualization feature
        def get_feature_priority(item):
            feature_name, sample_value = item

            # Highest priority: specified visualization_feature
            if visualization_feature and feature_name == visualization_feature:
                return (0, feature_name)

            # Second priority: features that will become video-encoded (images/visualizations)
            feature_type = FeatureType.from_data(sample_value)
            target_codec = codec_config.get_codec_for_feature(
                feature_type, feature_name)
            container_codec = codec_config.get_container_codec(target_codec)
            if container_codec in {"ffv1", "libaom-av1", "libx264", "libx265"}:
                return (1, feature_name)

            # Third priority: everything else
            return (2, feature_name)

        # Sort features by priority
        sorted_features = sorted(flattened_data.items(),
                                 key=get_feature_priority)

        feature_to_stream_idx = {}

        for feature_name, sample_value in sorted_features:
            # Determine feature type from sample
            feature_type = FeatureType.from_data(sample_value)

            # Determine optimal codec for this feature
            target_codec = codec_config.get_codec_for_feature(
                feature_type, feature_name)
            container_codec = codec_config.get_container_codec(target_codec)

            # Create stream with target codec directly
            stream = self.add_stream_for_feature(
                feature_name=feature_name,
                feature_type=feature_type,
                codec_config=codec_config,
                encoding=container_codec,
            )

            feature_to_stream_idx[feature_name] = stream.index

            logger.debug(
                f"Created stream for '{feature_name}' with codec '{container_codec}' (target: '{target_codec}') at index {stream.index}"
            )

        return feature_to_stream_idx

    def encode_batch_data_directly(
        self,
        data_batch: List[Dict[str, Any]],
        feature_to_stream_idx: Dict[str, int],
        codec_config: Any,
        feature_name_separator: str = "/",
        fps: Union[int, Dict[str, int]] = 10,
    ) -> None:
        """Encode a batch of data directly to target codecs without intermediate transcoding.

        Args:
            data_batch: List of data dictionaries
            feature_to_stream_idx: Mapping of feature names to stream indices
            codec_config: Codec configuration
            feature_name_separator: Separator for nested feature names
            fps: Frames per second for timestamp calculation. Can be an int (same fps for all features) or Dict[str, int] (per-feature fps)
        """
        from robodm.utils.flatten import _flatten_dict

        # Handle fps parameter - can be int or dict
        if isinstance(fps, int):
            # Use same fps for all features
            default_fps = fps
            feature_fps = {}
        else:
            # Per-feature fps specified
            feature_fps = fps
            default_fps = 10  # Fallback default

        # Initialize per-feature timestamps and time intervals
        feature_timestamps = {}
        feature_time_intervals = {}

        # Get all feature names from first sample to initialize timestamps
        if data_batch:
            first_sample = _flatten_dict(data_batch[0],
                                         sep=feature_name_separator)
            for feature_name in first_sample.keys():
                if feature_name in feature_to_stream_idx:
                    fps_for_feature = feature_fps.get(feature_name,
                                                      default_fps)
                    feature_timestamps[feature_name] = 0
                    feature_time_intervals[
                        feature_name] = 1000.0 / fps_for_feature

        for step_data in data_batch:
            flattened_data = _flatten_dict(step_data,
                                           sep=feature_name_separator)

            for feature_name, value in flattened_data.items():
                if feature_name in feature_to_stream_idx:
                    stream_idx = feature_to_stream_idx[feature_name]

                    # Get current timestamp for this feature
                    current_timestamp = feature_timestamps.get(feature_name, 0)

                    # Encode directly to target format
                    packet_infos = self.encode_data_to_packets(
                        data=value,
                        stream_index=stream_idx,
                        timestamp=int(current_timestamp),
                        codec_config=codec_config,
                        force_direct_encoding=True,
                    )

                    # Mux packets immediately
                    for packet_info in packet_infos:
                        self.mux_packet_info(packet_info)

                    # Update timestamp for this feature
                    time_interval = feature_time_intervals.get(
                        feature_name, 1000.0 / default_fps)
                    feature_timestamps[feature_name] = int(current_timestamp +
                                                           time_interval)
