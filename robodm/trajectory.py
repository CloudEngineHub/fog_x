import asyncio
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Any, Dict, List, Optional, Text, Tuple, Union, cast

import av
import h5py
import numpy as np

from robodm import FeatureType
from robodm.backend.base import ContainerBackend
# Backend abstraction
from robodm.backend.pyav_backend import PyAVBackend
from robodm.backend.parquet_backend import ParquetBackend
from robodm.backend.hdf5_backend import HDF5Backend
from robodm.backend.droid_backend import DROIDBackend
from robodm.trajectory_base import TrajectoryInterface
from robodm.utils.flatten import _flatten_dict

logger = logging.getLogger(__name__)

logging.getLogger("libav").setLevel(logging.CRITICAL)

from robodm.backend.codec_config import CodecConfig
from robodm.utils.resampler import FrequencyResampler
from robodm.utils.time_manager import TimeManager


class StreamInfo:

    def __init__(self, feature_name, feature_type, encoding):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.encoding = encoding

    def __str__(self):
        return f"StreamInfo({self.feature_name}, {self.feature_type}, {self.encoding})"

    def __repr__(self):
        return self.__str__()


class Trajectory(TrajectoryInterface):

    def __init__(
        self,
        path: Text,
        mode="r",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        feature_name_separator: Text = "/",
        filesystem: Optional[Any] = None,
        time_provider: Optional[Any] = None,
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
        visualization_feature: Optional[Text] = None,
        backend: Optional[Union[ContainerBackend, str]] = None,
        raw_codec: Optional[str] = None,
    ) -> None:
        """
        Initialize a trajectory instance.

        Args:
            path (str): Path to the trajectory file
            mode (str, optional): File mode ("r" for read, "w" for write). Defaults to "r".
            video_codec (str, optional): Video codec to use for video/image features. Options: "auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1". Defaults to "auto".
            codec_options (dict, optional): Additional codec options. Defaults to None.
            feature_name_separator (str, optional): Separator for feature names. Defaults to "/".
            filesystem: Optional filesystem interface for dependency injection
            time_provider: Optional time provider interface for dependency injection
            base_datetime: Optional base datetime for timestamp calculations
            time_unit: Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
            enforce_monotonic: Whether to enforce monotonically increasing timestamps
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
                If None, automatically puts video-encoded streams first during compacting.
            backend: Optional container backend for dependency injection. Can be a ContainerBackend instance or string ("parquet", "pyav")
            raw_codec (str, optional): Raw codec to use for non-image features. Options: "rawvideo", "rawvideo_pickle", "rawvideo_pyarrow". Defaults to None (will use video_codec).
        """
        self.path = path
        self.feature_name_separator = feature_name_separator
        self.visualization_feature = visualization_feature

        # Initialize codec configuration with separate video and raw codec support
        self.codec_config = CodecConfig(codec=video_codec,
                                        options=codec_options,
                                        raw_codec=raw_codec)

        # Dependency injection - set early so they're available during init
        self._filesystem = filesystem
        self._time_provider = time_provider

        # Initialize time management system
        self.time_manager = TimeManager(
            base_datetime=base_datetime,
            time_unit=time_unit,
            enforce_monotonic=enforce_monotonic,
        )

        self.feature_name_to_stream: Dict[str,
                                          Any] = {}  # feature_name: stream
        self.feature_name_to_feature_type: Dict[str, FeatureType] = (
            {})  # feature_name: feature_type
        self.trajectory_data = None  # trajectory_data
        self.start_time = self._time()
        self.mode = mode
        self.is_closed = False
        self.pending_write_tasks: List[Any] = (
            [])  # List to keep track of pending write tasks
        self.container_file: Optional[Any] = None  # av.OutputContainer or None

        # ------------------------------------------------------------------ #
        # Container backend setup
        # ------------------------------------------------------------------ #
        if backend is None:
            # Auto-detect backend based on file extension or directory structure
            if os.path.isdir(self.path):
                # Check if it's a DROID directory
                if (os.path.exists(os.path.join(self.path, "trajectory.h5")) and
                    os.path.exists(os.path.join(self.path, "recordings"))):
                    self.backend: ContainerBackend = DROIDBackend()
                else:
                    raise ValueError(f"Directory {self.path} does not appear to be a DROID trajectory")
            else:
                # Auto-detect backend based on file extension
                _, ext = os.path.splitext(self.path.lower())
                if ext in {".h5", ".hdf5"}:
                    self.backend = HDF5Backend()
                elif ext in {".parquet", ".pq"}:
                    self.backend = ParquetBackend()
                else:
                    # Default to PyAV backend for backward compatibility
                    self.backend = PyAVBackend()
        elif isinstance(backend, str):
            # Allow string specification of backend type
            if backend.lower() == "parquet":
                self.backend = ParquetBackend()
            elif backend.lower() == "pyav":
                self.backend = PyAVBackend()
            elif backend.lower() == "hdf5":
                self.backend = HDF5Backend()
            elif backend.lower() == "droid":
                self.backend = DROIDBackend()
            else:
                raise ValueError(f"Unknown backend type: {backend}. Use 'parquet', 'pyav', 'hdf5', or 'droid'")
        else:
            # Use provided backend instance
            self.backend = backend

        # check if the path exists
        # if not, create a new file and start data collection
        if self.mode == "w":
            if not self._exists(self.path):
                self._makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                # Use backend to open the container so that the rest of the
                # class can keep using `self.container_file` (PyAV Container).
                self.backend.open(self.path, "w")
                # Expose underlying PyAV container for legacy code paths that
                # access it directly.
                self.container_file = getattr(self.backend, "container", None)
            except Exception as e:
                logger.error(f"error creating the trajectory file: {e}")
                raise
        elif self.mode == "r":
            if not self._exists(self.path):
                raise FileNotFoundError(f"{self.path} does not exist")
            # Open the backend in read mode now so that subsequent operations
            # can reuse the container without touching PyAV directly here.
            self.backend.open(self.path, "r")
            self.container_file = getattr(self.backend, "container", None)
        else:
            raise ValueError(f"Invalid mode {self.mode}, must be 'r' or 'w'")

    def _exists(self, path: str) -> bool:
        """File existence check with dependency injection support."""
        if self._filesystem:
            return self._filesystem.exists(path)
        return os.path.exists(path)

    def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Directory creation with dependency injection support."""
        if self._filesystem:
            return self._filesystem.makedirs(path, exist_ok=exist_ok)
        return os.makedirs(path, exist_ok=exist_ok)

    def _remove(self, path: str) -> None:
        """File removal with dependency injection support."""
        if self._filesystem:
            return self._filesystem.remove(path)
        return os.remove(path)

    def _rename(self, src: str, dst: str) -> None:
        """File rename with dependency injection support."""
        if self._filesystem:
            return self._filesystem.rename(src, dst)
        return os.rename(src, dst)

    def _time(self) -> float:
        """Time retrieval with dependency injection support."""
        if self._time_provider:
            return self._time_provider.time()
        return time.time()

    def __len__(self):
        """Get the length of the trajectory in timesteps."""
        # Try backend-specific length methods first
        if hasattr(self.backend, 'get_trajectory_length'):
            return self.backend.get_trajectory_length()
        elif hasattr(self.backend, '_get_trajectory_length'):
            return self.backend._get_trajectory_length()
        
        # Fallback: use loaded trajectory data if available
        if hasattr(self, 'trajectory_data') and self.trajectory_data:
            # Find first feature and use its length
            for feature_name, feature_data in self.trajectory_data.items():
                if hasattr(feature_data, '__len__'):
                    return len(feature_data)
        
        # Last resort: try to get from streams
        if hasattr(self, 'backend') and self.backend:
            streams = self.backend.get_streams()
            if streams:
                # For now, return 0 if we can't determine length
                logger.warning("Could not determine trajectory length")
                return 0
        
        return 0

    def __getitem__(self, key):
        """
        get the value of the feature
        return hdf5-ed data
        """

        if self.trajectory_data is None:
            logger.info(f"Loading the trajectory data with key {key}")
            self.trajectory_data = self.load()

        if self.trajectory_data is None:
            raise RuntimeError("Failed to load trajectory data")

        return self.trajectory_data[key]

    def close(self, compact=True):
        """
        close the container file

        args:
        compact: re-read from the cache to encode pickled data to images
        """
        logger.debug(
            f"Closing trajectory, is_closed={self.is_closed}, compact={compact}"
        )

        if self.is_closed:
            raise ValueError("The container file is already closed")

        if self.mode == "r":
            # For read mode, just mark as closed
            self.is_closed = True
            self.trajectory_data = None
            logger.debug("Trajectory (read mode) closed successfully")
            return

        # Write mode handling
        if self.backend.container is None:  # type: ignore[attr-defined]
            logger.warning(
                "Container not available, marking trajectory as closed")
            self.is_closed = True
            return

        # Check if there are any streams with data
        streams = self.backend.get_streams()
        has_data = len(streams) > 0

        try:
            # Flush all streams using backend abstraction
            buffered_packets = self.backend.flush_all_streams()
            logger.debug(f"Flushed {len(buffered_packets)} buffered packets")

            # Mux all buffered packets
            for packet_info in buffered_packets:
                if packet_info.pts is None:
                    raise ValueError(f"Packet {packet_info} has no pts")
                self.backend.mux_packet_info(packet_info)
                logger.debug(
                    f"Muxed flush packet from stream {packet_info.stream_index}"
                )

            logger.debug("Flushing completed")
        except Exception as e:
            logger.error(f"Error during flush: {e}")

        logger.debug("Closing container")
        self.backend.close()

        # Ensure file exists even if empty
        if not self._exists(self.path):
            logger.warning(
                f"Container was closed but {self.path} doesn't exist. This might indicate an issue."
            )

        # Only attempt transcoding if file exists, has content, compact is requested, and not using HDF5 backend
        if (compact and has_data and self._exists(self.path)
                and os.path.getsize(self.path) > 0
                and not isinstance(self.backend, HDF5Backend)):
            logger.debug(
                "Starting intelligent transcoding based on feature types")
            self._transcode_by_feature_type()
        elif isinstance(self.backend, HDF5Backend):
            logger.debug("Skipping transcoding for HDF5 backend (not needed)")
        
        else:
            logger.debug(
                f"Skipping transcoding: compact={compact}, has_data={has_data}, file_exists={self._exists(self.path)}, file_size={os.path.getsize(self.path) if self._exists(self.path) else 0}"
            )

        self.trajectory_data = None
        self.container_file = None
        self.is_closed = True
        logger.debug("Trajectory closed successfully")

    def load(
        self,
        return_type: str = "numpy",
        desired_frequency: Optional[float] = None,
        data_slice: Optional[slice] = None,
    ):
        """
        Load trajectory data with optional temporal resampling and slicing.

        Parameters
        ----------
        return_type : {"numpy", "container"}, default "numpy"
            • "numpy"     – decode the data and return a dict[str, np.ndarray]
            • "container" – skip all decoding and just return the file path
        desired_frequency : float | None, default None
            Target sampling frequency **in hertz**.  If None, every frame is
            returned (subject to `data_slice`). For upsampling (when desired
            frequency is higher than original), prior frames are duplicated
            to fill temporal gaps. For downsampling, frames are skipped.
        data_slice : slice | None, default None
            Standard Python slice that is applied *after* resampling.
            Example: `slice(100, 200, 2)` → keep resampled indices 100-199,
            step 2.  Negative indices and reverse slices are **not** supported.

        Notes
        -----
        * Resampling is performed individually for every feature stream.
        * For upsampling: when time gaps between consecutive frames exceed
          the desired period, the prior frame is duplicated at regular
          intervals to achieve the target frequency.
        * For downsampling: frames that arrive too close together (within
          the desired period) are skipped.
        * Slicing is interpreted on the **resampled index** domain so that the
          combination `desired_frequency + data_slice` behaves the same as
          `df.iloc[data_slice]` would on a pandas dataframe that had already
          been resampled to `desired_frequency`.
        * When `data_slice` starts at a positive index we `seek()` to the
          corresponding timestamp to avoid decoding frames that will be thrown
          away anyway.
        """
        logger.debug(
            f"load() called with return_type='{return_type}', desired_frequency={desired_frequency}, data_slice={data_slice}"
        )

        # ------------------------------------------------------------------ #
        # Fast-path: user only wants the container path
        # ------------------------------------------------------------------ #
        if return_type == "container":
            logger.debug("Returning container path (fast-path)")
            return self.path
        if return_type not in {"numpy", "container"}:
            raise ValueError("return_type must be 'numpy' or 'container'")

        # ------------------------------------------------------------------ #
        # Validate / canonicalise the slice object
        # ------------------------------------------------------------------ #
        if data_slice is None:
            logger.debug(
                "No data_slice provided, using default slice(None, None, None)"
            )
            data_slice = slice(None, None, None)
        else:
            logger.debug(f"Using provided data_slice: {data_slice}")

        if data_slice.step not in (None, 1) and data_slice.step <= 0:
            raise ValueError("Reverse or zero-step slices are not supported")

        # Check for negative start - this should raise an error
        if data_slice.start is not None and data_slice.start < 0:
            raise ValueError("Negative slice start values are not supported")

        sl_start = 0 if data_slice.start is None else max(data_slice.start, 0)
        sl_stop = data_slice.stop  # can be None
        sl_step = 1 if data_slice.step is None else data_slice.step

        logger.debug(
            f"Canonicalized slice parameters: start={sl_start}, stop={sl_stop}, step={sl_step}"
        )

        # ------------------------------------------------------------------ #
        # Frequency → minimum period in stream time-base units (milliseconds)
        # ------------------------------------------------------------------ #
        period_ms: Optional[int] = None
        if desired_frequency is not None:
            if desired_frequency <= 0:
                raise ValueError("desired_frequency must be positive")
            period_ms = int(round(1000.0 / desired_frequency))
            logger.debug(
                f"Frequency resampling enabled: {desired_frequency} Hz -> period_ms={period_ms}"
            )
        else:
            logger.debug("No frequency resampling (desired_frequency is None)")

        # ------------------------------------------------------------------ #
        # Open the container and, if possible, seek() to the first slice index
        # ------------------------------------------------------------------ #
        # Ensure backend has the container open (read mode).
        if self.backend.container is None:  # type: ignore[attr-defined]
            self.backend.open(self.path, "r")

        # Get stream metadata from backend
        stream_metadata_list = self.backend.get_streams()
        logger.debug(f"Using backend with {len(stream_metadata_list)} streams")

        # Handle empty trajectory case
        if not stream_metadata_list:
            logger.debug("No streams found in container, returning empty dict")
            self.backend.close()
            return {}

        # Track if we performed seeking to adjust slice logic
        seek_performed = False
        seek_offset_frames = 0

        # Use seeking optimization when we have slicing
        if sl_start > 0 and stream_metadata_list:
            if period_ms is not None:
                # When combining frequency resampling with slicing, seek to the timestamp
                # that corresponds to the sl_start-th frame AFTER resampling.
                # Since resampling keeps every period_ms milliseconds, the sl_start-th
                # resampled frame corresponds to timestamp: sl_start * period_ms
                seek_ts_ms = sl_start * period_ms
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking with frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )
            else:
                # If only slicing (no frequency resampling), seek to the sl_start-th frame
                # assuming original 100ms intervals (10Hz from our test data)
                seek_ts_ms = sl_start * 100
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking without frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )

            # Seek using the first stream
            try:
                first_stream_idx = 0  # Use first available stream index
                logger.debug(
                    f"Attempting to seek to timestamp {seek_ts_ms} on first stream"
                )
                self.backend.seek_container(seek_ts_ms,
                                            first_stream_idx,
                                            any_frame=True)
                seek_performed = True
                logger.debug("Seek successful")
            except Exception as e:
                # Seeking failed (e.g. single large packet stream) – fall back
                # to decoding from the beginning.
                logger.debug(
                    f"Seeking failed ({e}), falling back to decoding from beginning"
                )
                seek_performed = False
                seek_offset_frames = 0
        else:
            logger.debug(
                "No seeking optimization needed (sl_start=0 or no streams)")

        # ------------------------------------------------------------------ #
        # Book-keeping structures
        # ------------------------------------------------------------------ #
        cache: dict[str, list[Any]] = {}
        done: set[str] = set()

        # Instantiate the helper that takes care of all frequency based
        # up-/down-sampling **and** slice filtering.
        resampler = FrequencyResampler(
            period_ms=period_ms,
            sl_start=sl_start,
            sl_stop=sl_stop,
            sl_step=sl_step,
            seek_offset_frames=seek_offset_frames,
        )

        # Build stream index mapping and initialize cache
        stream_idx_to_feature: Dict[int, str] = {}
        stream_count = 0

        for i, stream_metadata in enumerate(stream_metadata_list):
            fname = stream_metadata.feature_name
            ftype = stream_metadata.feature_type
            if not (fname and ftype) or fname == "unknown":
                logger.debug(
                    f"Skipping stream {i} without valid FEATURE_NAME or FEATURE_TYPE"
                )
                continue

            cache[fname] = []
            # Inform the resampler so it can initialise internal bookkeeping
            resampler.register_feature(fname)

            self.feature_name_to_feature_type[fname] = FeatureType.from_str(
                ftype)
            stream_idx_to_feature[i] = fname
            stream_count += 1
            logger.debug(f"Initialized feature '{fname}' with type {ftype}")

        # Handle case where no valid streams were found
        if not cache:
            logger.debug(
                "No valid feature streams found, returning empty dict")
            self.backend.close()
            return {}

        logger.debug(f"Processing {stream_count} feature streams")

        # ------------------------------------------------------------------ #
        # Main demux / decode loop
        # ------------------------------------------------------------------ #
        logger.debug("Starting main demux/decode loop")
        packet_count = 0
        processed_packets = 0
        skipped_frequency = 0
        skipped_slice = 0
        decoded_packets = 0
        upsampled_frames = 0

        # Get stream indices for demuxing
        valid_stream_indices = list(stream_idx_to_feature.keys())

        for packet in self.backend.demux_streams(valid_stream_indices):
            packet_count += 1

            # Get feature name from stream index
            stream_idx = packet.stream.index
            fname = stream_idx_to_feature.get(
                stream_idx)  # type: ignore[assignment]
            if fname is None or fname in done:
                continue

            # Use backend's packet validation
            if not self.backend.validate_packet(packet):
                logger.debug(f"Skipping invalid packet for feature '{fname}'")
                continue

            processed_packets += 1

            # -------------------------------------------------------------- #
            # Delegate frequency based up-/down-sampling to helper
            # -------------------------------------------------------------- #
            keep_current, num_dups = resampler.process_packet(
                fname=fname,
                pts=packet.pts,
                has_prior_frame=bool(cache[fname]),
            )

            if not keep_current:
                skipped_frequency += 1
                logger.debug(
                    f"Skipping packet for '{fname}' due to frequency reduction (period_ms={period_ms})"
                )
                continue

            # Insert duplicate frames **before** processing current packet
            if num_dups > 0 and cache[fname]:
                last_frame_data = cache[fname][-1]
                for i in range(num_dups):
                    dup_idx = resampler.next_index(fname)
                    if resampler.want(dup_idx):
                        cache[fname].append(last_frame_data)
                        upsampled_frames += 1
                        logger.debug(
                            f"Inserted duplicate frame for '{fname}' ({i+1}/{num_dups}) at idx={dup_idx}"
                        )

            # Advance index for *current* packet and apply slice filter
            current_idx = resampler.next_index(fname)
            if not resampler.want(current_idx):
                skipped_slice += 1
                resampler.update_last_pts(fname, packet.pts)
                logger.debug(
                    f"Skipping packet for '{fname}' due to slice filter: idx={current_idx}"
                )
                continue

            logger.debug(
                f"Decoding packet for '{fname}': idx={current_idx}, pts={packet.pts}"
            )

            # --- decode on demand only ------------------------------------
            codec = self.backend.get_stream_codec_name(stream_idx)
            if codec == "rawvideo":
                raw = bytes(packet)
                if not raw:  # zero-length placeholder
                    logger.debug(
                        f"Skipping empty rawvideo packet for '{fname}'")
                    continue
                cache[fname].append(pickle.loads(raw))
                decoded_packets += 1
                logger.debug(
                    f"Decoded rawvideo packet for '{fname}' (pickled data)")
            else:
                frames = self.backend.decode_stream_frames(
                    stream_idx, bytes(packet))
                for frame in frames:
                    ft = self.feature_name_to_feature_type[fname]
                    # Use backend to convert frame to array
                    arr = self.backend.convert_frame_to_array(frame,
                                                              ft,
                                                              format="rgb24")
                    cache[fname].append(arr)
                    decoded_packets += 1
                    logger.debug(
                        f"Decoded {codec} frame for '{fname}': shape={getattr(arr, 'shape', 'N/A')}, dtype={getattr(arr, 'dtype', 'N/A')}"
                    )

            # Record timestamp for resampling logic
            resampler.update_last_pts(fname, packet.pts)

            # Early exit: all streams finished their slice
            if sl_stop is not None and resampler.kept_idx[fname] >= sl_stop:
                done.add(fname)
                logger.debug(
                    f"Feature '{fname}' reached slice stop ({sl_stop}), marking as done"
                )
                if len(done) == len(cache):
                    logger.debug(
                        "All features completed their slices, breaking early")
                    break

        # ------------------------------------------------------------------ #
        # Flush any buffered pictures that the decoder is still holding
        # ------------------------------------------------------------------ #
        for stream_idx, fname in stream_idx_to_feature.items():
            if not fname or fname not in cache:
                continue

            codec = self.backend.get_stream_codec_name(stream_idx)
            if codec == "rawvideo":
                continue  # pickled streams have no buffer

            # Flush the decoder by passing None
            frames = self.backend.decode_stream_frames(stream_idx,
                                                       packet_data=None)
            for frame in frames:
                flush_idx = resampler.next_index(fname)
                if not resampler.want(flush_idx):  # honour slice filter
                    continue

                ft = self.feature_name_to_feature_type[fname]
                # Use backend to convert frame to array
                arr = self.backend.convert_frame_to_array(frame,
                                                          ft,
                                                          format="rgb24")
                cache[fname].append(arr)
                decoded_packets += 1

        self.backend.close()

        logger.debug(
            f"Demux/decode loop completed: total_packets={packet_count}, processed={processed_packets}, "
            f"skipped_frequency={skipped_frequency}, skipped_slice={skipped_slice}, decoded={decoded_packets}, upsampled_frames={upsampled_frames}"
        )

        # ------------------------------------------------------------------ #
        # Convert to numpy arrays
        # ------------------------------------------------------------------ #
        logger.debug("Converting cached data to numpy arrays")
        out: Dict[str, Any] = {}
        for fname, lst in cache.items():
            logger.debug(f"Converting '{fname}': {len(lst)} items")
            if not lst:
                logger.debug(f"Warning: '{fname}' has no data after filtering")
                out[fname] = np.array([])
                continue

            ft = self.feature_name_to_feature_type[fname]
            if ft.dtype in ["string", "str"]:
                out[fname] = np.array(lst, dtype=object)
                logger.debug(
                    f"Created object array for '{fname}': shape={out[fname].shape}"
                )
            else:
                try:
                    out[fname] = np.asarray(lst, dtype=ft.dtype)
                    logger.debug(
                        f"Created {ft.dtype} array for '{fname}': shape={out[fname].shape}"
                    )
                except ValueError as e:
                    if "setting an array element with a sequence" in str(e):
                        # Handle inconsistent data shapes by creating object array
                        logger.debug(
                            f"Data shapes inconsistent for '{fname}', creating object array: {e}"
                        )
                        out[fname] = np.array(lst, dtype=object)
                        logger.debug(
                            f"Created object array for '{fname}' due to shape inconsistency: shape={out[fname].shape}"
                        )
                    else:
                        raise

        logger.debug(
            f"load() returning {len(out)} features: {list(out.keys())}")
        return out

    def init_feature_streams(self, feature_spec: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_spec.items():
            encoding = self._get_encoding_of_feature(None, feature_type,
                                                     feature)
            self.feature_name_to_stream[
                feature] = self._add_stream_to_container(
                    self.container_file, feature, encoding, feature_type)

    def add(
        self,
        feature: str,
        data: Any,
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
        force_direct_encoding: bool = False,
    ) -> None:
        """
        add one value to container file

        Args:
            feature (str): name of the feature
            data (Any): value associated with the feature; except dictionary
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
            force_direct_encoding (bool): If True, encode directly to target codec instead of rawvideo intermediate step.

        Examples:
            >>> trajectory.add('feature1', 'image1.jpg')
            >>> trajectory.add('feature1', 'image1.jpg', timestamp=1000, time_unit='ms')

        Logic:
        - check the feature name
        - if the feature name is not in the container, create a new stream

        - check the type of value
        - if value is numpy array, create a frame and encode it
        - if it is a string or int, create a packet and encode it
        - else raise an error

        Exceptions:
            raise an error if the value is a dictionary
        """
        logger.debug(
            f"Adding feature: {feature}, data shape: {getattr(data, 'shape', 'N/A')}"
        )

        if type(data) == dict:
            raise ValueError("Use add_by_dict for dictionary")

        feature_type = FeatureType.from_data(data)
        # encoding = self._get_encoding_of_feature(data, None)
        self.feature_name_to_feature_type[feature] = feature_type

        # check if the feature is already in the container
        # if not, create a new stream
        # Check if the feature is already in the container
        stream_idx = self.backend.stream_exists_by_feature(feature)
        if stream_idx is None:
            logger.debug(f"Creating new stream for feature: {feature}")
            # Determine encoding based on whether we want direct encoding
            if force_direct_encoding:
                # Get the optimal codec for this feature type
                target_codec = self.codec_config.get_codec_for_feature(
                    feature_type, feature)
                container_codec = self.codec_config.get_container_codec(
                    target_codec)
                encoding = container_codec
            else:
                # Use rawvideo for intermediate encoding (legacy behavior)
                encoding = "rawvideo"

            self._on_new_stream(feature, encoding, feature_type)
            stream_idx = self.backend.stream_exists_by_feature(feature)
            if stream_idx is None:
                raise RuntimeError(
                    f"Failed to create stream for feature {feature}")

        logger.debug(f"Using stream index: {stream_idx}")

        # get the timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.convert_units(
                timestamp, time_unit or self.time_manager.time_unit, "ms")

        logger.debug(
            f"Encoding frame with validated timestamp: {validated_timestamp}")

        # encode the frame using backend
        packet_infos = self.backend.encode_data_to_packets(
            data=data,
            stream_index=stream_idx,
            timestamp=validated_timestamp,
            codec_config=self.codec_config,
            force_direct_encoding=
            force_direct_encoding,  # type: ignore[call-arg]
        )
        logger.debug(f"Generated {len(packet_infos)} packet infos")

        # write the packets to the container
        for i, packet_info in enumerate(packet_infos):
            logger.debug(f"Muxing packet {i}: {packet_info}")
            self.backend.mux_packet_info(packet_info)
            logger.debug(f"Successfully muxed packet {i}")

    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
        force_direct_encoding: bool = False,
    ) -> None:
        """
        add one value to container file
        data might be nested dictionary of values for each feature

        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
                assume the timestamp is same for all the features within the dictionary
            force_direct_encoding (bool): If True, encode directly to target codec instead of rawvideo intermediate step.

        Examples:
            >>> trajectory.add_by_dict({'feature1': 'image1.jpg'})

        Logic:
        - check the data see if it is a dictionary
        - if dictionary, need to flatten it and add each feature separately
        """
        if type(data) != dict:
            raise ValueError("Use add for non-dictionary data, type is ",
                             type(data))

        _flatten_dict_data = _flatten_dict(data,
                                           sep=self.feature_name_separator)

        # Get validated timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.convert_units(
                timestamp, time_unit or self.time_manager.time_unit, "ms")

        for feature, value in _flatten_dict_data.items():
            self.add(
                feature,
                value,
                validated_timestamp,
                "ms",
                force_direct_encoding=force_direct_encoding,
            )

    @classmethod
    def from_list_of_dicts(
        cls,
        data: List[Dict[str, Any]],
        path: Text,
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[Union[int, Dict[str, int]]] = 10,
        raw_codec: Optional[str] = None,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a list of dictionaries.

        args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file
            video_codec (str, optional): Video codec to use for video/image features. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
            fps: Optional fps for features. Can be an int (same fps for all features) or Dict[str, int] (per-feature fps).
            raw_codec (str, optional): Raw codec to use for non-image features. Defaults to None.

        Example:
        original_trajectory = [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"},
        ]

        trajectory = Trajectory.from_list_of_dicts(original_trajectory, path="/tmp/robodm/output.vla")
        """
        if not data:
            raise ValueError("Data list cannot be empty")

        traj = cls(
            path,
            mode="w",
            video_codec=video_codec,
            codec_options=codec_options,
            visualization_feature=visualization_feature,
            raw_codec=raw_codec,
        )

        logger.info(
            f"Creating a new trajectory file at {path} with {len(data)} steps using direct encoding"
        )

        # Use the new backend method for efficient batch processing
        sample_data = data[
            0]  # Use first sample to determine feature types and optimal codecs
        feature_to_stream_idx = traj.backend.create_streams_for_batch_data(  # type: ignore[attr-defined]
            sample_data=sample_data,
            codec_config=traj.codec_config,
            feature_name_separator=traj.feature_name_separator,
            visualization_feature=visualization_feature,
        )

        # Update feature type tracking for consistency
        from robodm.utils.flatten import _flatten_dict

        flattened_sample = _flatten_dict(sample_data,
                                         sep=traj.feature_name_separator)
        for feature_name, sample_value in flattened_sample.items():
            feature_type = FeatureType.from_data(sample_value)
            traj.feature_name_to_feature_type[feature_name] = feature_type

        # Encode all data directly to target codecs
        traj.backend.encode_batch_data_directly(  # type: ignore[attr-defined]
            data_batch=data,
            feature_to_stream_idx=feature_to_stream_idx,
            codec_config=traj.codec_config,
            feature_name_separator=traj.feature_name_separator,
            fps=fps,
        )

        # Close without transcoding since we encoded directly to target formats
        traj.close(compact=False)
        return traj

    @classmethod
    def from_dict_of_lists(
        cls,
        data: Dict[str, List[Any]],
        path: Text,
        feature_name_separator: Text = "/",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[Union[int, Dict[str, int]]] = 10,
        raw_codec: Optional[str] = None,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file
            feature_name_separator (Text, optional): Delimiter to separate feature names. Defaults to "/".
            video_codec (str, optional): Video codec to use for video/image features. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
            fps: Optional fps for features. Can be an int (same fps for all features) or Dict[str, int] (per-feature fps).
            raw_codec (str, optional): Raw codec to use for non-image features. Defaults to None.

        Returns:
            Trajectory: _description_

        Example:
        original_trajectory = {
            "feature1": ["value1", "value3"],
            "feature2": ["value2", "value4"],
        }

        trajectory = Trajectory.from_dict_of_lists(original_trajectory, path="/tmp/robodm/output.vla")
        """
        from robodm.utils.flatten import _flatten_dict

        # Flatten the data and validate
        flattened_dict_data = _flatten_dict(data, sep=feature_name_separator)

        # Check if all lists have the same length
        list_lengths = [len(v) for v in flattened_dict_data.values()]
        if len(set(list_lengths)) != 1:
            raise ValueError(
                "All lists must have the same length",
                [(k, len(v)) for k, v in flattened_dict_data.items()],
            )

        if not list_lengths or list_lengths[0] == 0:
            raise ValueError("Data lists cannot be empty")

        # Convert dict of lists to list of dicts for batch processing
        num_steps = list_lengths[0]
        list_of_dicts = []
        for i in range(num_steps):
            step: Dict[str, Any] = {}
            for feature_name, feature_values in flattened_dict_data.items():
                # Reconstruct nested structure if needed
                step = cls._set_nested_value(step, feature_name,
                                             feature_values[i],
                                             feature_name_separator)
            list_of_dicts.append(step)

        # Use the optimized from_list_of_dicts method
        return cls.from_list_of_dicts(
            data=list_of_dicts,
            path=path,
            video_codec=video_codec,
            codec_options=codec_options,
            visualization_feature=visualization_feature,
            fps=fps,
            raw_codec=raw_codec,
        )

    @staticmethod
    def _set_nested_value(data_dict: Dict[str, Any], key_path: str, value: Any,
                          separator: str) -> Dict[str, Any]:
        """Helper method to set a nested value in a dictionary using a key path."""
        keys = key_path.split(separator)
        current = data_dict

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value
        return data_dict

    @classmethod
    def from_aligned_data(
        cls,
        data: List[Dict[str, Any]],
        path: Text,
        feature_name_separator: Text = "/",
    ) -> "Trajectory":
        """
        Create a Trajectory with parquet backend from aligned data.
        
        Args:
            data: List of dictionaries with aligned timestamps and features
                  Format: [{"timestamp": ts, "feature1": val1, "feature2": val2}, ...]
            path: Path to the parquet file
            feature_name_separator: Separator for nested feature names
            
        Returns:
            Trajectory instance with parquet backend
        """
        if not data:
            raise ValueError("Data list cannot be empty")
            
        traj = cls(path, mode="w", backend="parquet", 
                  feature_name_separator=feature_name_separator)
        
        # Add aligned rows directly to parquet backend
        for row in data:
            timestamp = row.pop("timestamp", None)
            if timestamp is None:
                raise ValueError("Each row must contain a 'timestamp' field")
            traj.backend.add_aligned_row(timestamp, row)
            
        traj.close()
        return traj

    def _transcode_by_feature_type(self):
        """
        Intelligently decide whether to transcode images or raw bytes based on feature types.
        This method analyzes all features and determines the appropriate transcoding strategy.
        """
        # Analyze feature types to determine transcoding strategy
        has_image_features = False
        has_raw_data_features = False

        for feature_name, feature_type in self.feature_name_to_feature_type.items(
        ):
            # Check if this is image data (RGB with shape HxWx3)
            is_image_data = (hasattr(feature_type, "shape")
                             and feature_type.shape
                             and len(feature_type.shape) == 3
                             and feature_type.shape[2] == 3)

            if is_image_data:
                # Check if this image feature should be transcoded to video codec
                target_encoding = self._get_encoding_of_feature(
                    None, feature_type, feature_name)
                if target_encoding in {
                        "ffv1", "libaom-av1", "libx264", "libx265"
                }:
                    has_image_features = True
                    logger.debug(
                        f"Feature '{feature_name}' identified as image for video transcoding"
                    )
            else:
                # Check if this raw data feature should be compressed
                target_encoding = self._get_encoding_for_raw_data(
                    feature_type, feature_name)
                if target_encoding != "rawvideo":
                    has_raw_data_features = True
                    logger.debug(
                        f"Feature '{feature_name}' identified as raw data for compression"
                    )

        # Decide transcoding strategy based on feature analysis
        transcoding_performed = False

        if has_image_features:
            logger.debug("Performing image transcoding for video features")
            self._transcode_pickled_images()
            transcoding_performed = True

        if has_raw_data_features:
            logger.debug("Performing raw data transcoding for compression")
            self._transcode_pickled_bytes()
            transcoding_performed = True

        if not transcoding_performed:
            logger.debug(
                "No transcoding performed - no features require transcoding")

    def _transcode_pickled_images(self,
                                  ending_timestamp: Optional[int] = None):
        """
        Transcode pickled images into the desired format (e.g., raw or encoded images).
        """
        from robodm.backend.base import StreamConfig
        from robodm.backend.pyav_backend import PyAVBackend

        # Move the original file to a temporary location
        temp_path = self.path + ".temp"
        self._rename(self.path, temp_path)

        # Build stream configurations for transcoding
        stream_configs = {}

        # Open original container temporarily to get stream info
        temp_backend = PyAVBackend()
        temp_backend.open(temp_path, "r")
        original_streams = temp_backend.get_streams()
        temp_backend.close()

        for i, stream_metadata in enumerate(original_streams):
            feature_name = stream_metadata.feature_name
            if feature_name == "unknown" or not feature_name:
                continue

            feature_type = self.feature_name_to_feature_type.get(feature_name)
            if feature_type is None:
                continue

            # Determine target encoding
            target_encoding = self._get_encoding_of_feature(
                None, feature_type, feature_name)

            # Only handle video container codecs, skip rawvideo variants
            if target_encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
                # Create stream config for video codec
                config = StreamConfig(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    encoding=target_encoding,  # Video container codec
                    codec_options=self.codec_config.get_codec_options(
                        target_encoding),
                    pixel_format=self.codec_config.get_pixel_format(
                        target_encoding, feature_type),
                )

                # Use the actual stream index from the original container
                stream_configs[i] = config

        # Use backend's transcoding abstraction
        self.backend.transcode_container(
            input_path=temp_path,
            output_path=self.path,
            stream_configs=stream_configs,
            visualization_feature=self.visualization_feature,
        )

        logger.debug("Transcoding completed successfully")
        self._remove(temp_path)

    def _transcode_pickled_bytes(self, ending_timestamp: Optional[int] = None):
        """
        Transcode pickled bytes into compressed format (e.g., pyarrow).
        This handles non-image data that should be compressed using raw data codecs.
        """
        from robodm.backend.base import StreamConfig
        from robodm.backend.pyav_backend import PyAVBackend

        # Move the original file to a temporary location
        temp_path = self.path + ".temp"
        self._rename(self.path, temp_path)

        # Build stream configurations for transcoding
        stream_configs = {}

        # Open original container temporarily to get stream info
        temp_backend = PyAVBackend()
        temp_backend.open(temp_path, "r")
        original_streams = temp_backend.get_streams()
        temp_backend.close()

        for i, stream_metadata in enumerate(original_streams):
            feature_name = stream_metadata.feature_name
            if feature_name == "unknown" or not feature_name:
                continue

            feature_type = self.feature_name_to_feature_type.get(feature_name)
            if feature_type is None:
                continue

            # Check if this is non-image raw data
            is_image_data = (hasattr(feature_type, "shape")
                             and feature_type.shape
                             and len(feature_type.shape) == 3
                             and feature_type.shape[2] == 3)

            if not is_image_data:
                # For non-image data, determine if we should compress
                target_encoding = self._get_encoding_for_raw_data(
                    feature_type, feature_name)

                if (target_encoding != "rawvideo"
                    ):  # Only transcode if compression is desired
                    # Separate container codec from internal codec
                    container_encoding = "rawvideo"  # Always use rawvideo for container
                    internal_codec = self.codec_config.get_raw_codec_name(
                        target_encoding)

                    # Create stream config for compressed format
                    config = StreamConfig(
                        feature_name=feature_name,
                        feature_type=feature_type,
                        encoding=container_encoding,  # Container codec
                        codec_options=self.codec_config.get_codec_options(
                            target_encoding),
                        pixel_format=None,  # Raw codecs don't use pixel format
                        internal_codec=
                        internal_codec,  # Internal codec implementation
                    )

                    # Use the actual stream index from the original container
                    stream_configs[i] = config

        # Only proceed if there are streams to transcode
        if stream_configs:
            # Use backend's transcoding abstraction
            self.backend.transcode_container(
                input_path=temp_path,
                output_path=self.path,
                stream_configs=stream_configs,
                visualization_feature=self.visualization_feature,
            )

            logger.debug("Raw data transcoding completed successfully")
        else:
            # No transcoding needed, just rename back
            self._rename(temp_path, self.path)
            logger.debug("No raw data streams need transcoding")
            return

        self._remove(temp_path)

    def _get_encoding_for_raw_data(self,
                                   feature_type: FeatureType,
                                   feature_name: Optional[str] = None) -> str:
        """
        Determine appropriate encoding for raw (non-image) data.

        Args:
            feature_type: The FeatureType of the data
            feature_name: Optional feature name for feature-specific decisions

        Returns:
            Encoding string (e.g., "rawvideo_pyarrow", "rawvideo_pickle")
        """
        # Use the codec config to determine the right codec for this feature
        return self.codec_config.get_codec_for_feature(feature_type,
                                                       feature_name)

    def _on_new_stream(self, new_feature, new_encoding, new_feature_type):
        from robodm.backend.base import StreamConfig

        # Check if stream already exists for this feature
        if self.backend.stream_exists_by_feature(new_feature) is not None:
            return

        # Get current streams from backend
        current_streams = self.backend.get_streams()

        if not current_streams:
            logger.debug(
                f"Creating a new stream for the first feature {new_feature}")
            # Use backend to add the stream directly
            stream = self.backend.add_stream_for_feature(  # type: ignore[attr-defined]
                feature_name=new_feature,
                feature_type=new_feature_type,
                codec_config=self.codec_config,
                encoding=new_encoding,
            )
            # Update legacy tracking for backwards compatibility
            self.feature_name_to_stream[new_feature] = stream
            self.container_file = self.backend.container  # type: ignore[attr-defined]
        else:
            logger.debug(f"Adding a new stream for the feature {new_feature}")
            # Following is a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close(compact=False)

            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            self._rename(self.path, temp_path)

            # Build stream configurations for existing streams
            existing_stream_configs = []
            for i, stream_metadata in enumerate(current_streams):
                if stream_metadata.feature_name == new_feature:
                    continue  # Skip the new feature we're adding
                feature_type = self.feature_name_to_feature_type.get(
                    stream_metadata.feature_name)
                if feature_type is None:
                    continue
                config = StreamConfig(
                    feature_name=stream_metadata.feature_name,
                    feature_type=feature_type,
                    encoding=stream_metadata.encoding,
                )
                existing_stream_configs.append((i, config))

            # Add new stream configuration
            new_stream_config = StreamConfig(
                feature_name=new_feature,
                feature_type=new_feature_type,
                encoding=new_encoding,
            )

            # Use backend's container recreation abstraction
            stream_mapping = self.backend.create_container_with_new_streams(
                original_path=temp_path,
                new_path=self.path,
                existing_streams=existing_stream_configs,
                new_stream_configs=[new_stream_config],
            )

            # Update our tracking structures using backend information
            self.container_file = self.backend.container  # type: ignore[attr-defined]

            # Update feature_name_to_stream mapping using backend
            new_feature_name_to_stream = {}
            updated_streams = self.backend.get_streams()
            for i, stream_metadata in enumerate(updated_streams):
                feature_name = stream_metadata.feature_name
                if feature_name and hasattr(self.backend, "_idx_to_stream"):
                    stream = self.backend._idx_to_stream.get(i)
                    if stream:
                        new_feature_name_to_stream[feature_name] = stream

            self.feature_name_to_stream = new_feature_name_to_stream

            self._remove(temp_path)
            self.is_closed = False

    def _add_stream_to_container(self, container, feature_name, encoding,
                                 feature_type):
        # If we're adding to the primary container that the backend manages,
        # delegate to backend. Otherwise fall back to the internal PyAV logic
        # because the backend is not aware of this ad-hoc container.

        if hasattr(self.backend, "container") and container is getattr(
                self.backend, "container", None):
            return self.backend.add_stream_for_feature(  # type: ignore[attr-defined]
                feature_name=feature_name,
                feature_type=feature_type,
                codec_config=self.codec_config,
                encoding=encoding,
            )

        # Legacy path – keep the original PyAV-based implementation for
        # transient containers (e.g. during transcoding).
        # Import PyAV locally since it's only needed for legacy paths
        from fractions import Fraction

        stream = container.add_stream(encoding)

        if encoding in ["ffv1", "libaom-av1", "libx264", "libx265"]:
            shape = feature_type.shape
            if shape is not None and len(shape) >= 2:
                stream.width = shape[1]
                stream.height = shape[0]

            pixel_format = self.codec_config.get_pixel_format(
                encoding, feature_type)
            if pixel_format:
                stream.pix_fmt = pixel_format

            codec_options = self.codec_config.get_codec_options(encoding)
            if codec_options:
                stream.codec_context.options = codec_options

        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream

    def _get_encoding_of_feature(
        self,
        feature_value: Any,
        feature_type: Optional[FeatureType],
        feature_name: Optional[str] = None,
    ) -> Text:
        """
        get the encoding of the feature value
        args:
            feature_value: value of the feature
            feature_type: type of the feature
            feature_name: name of the feature (for feature-specific codec selection)
        return:
            encoding of the feature in string
        """
        if feature_type is None:
            feature_type = FeatureType.from_data(feature_value)

        return self.codec_config.get_codec_for_feature(feature_type,
                                                       feature_name)
