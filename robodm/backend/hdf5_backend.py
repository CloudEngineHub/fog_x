"""HDF5-backed implementation of the ContainerBackend interface.

This module provides a native HDF5 storage backend for RoboDM trajectories,
offering efficient hierarchical data storage with direct access to structured
data without video encoding overhead.

The HDF5 backend maps RoboDM concepts to HDF5 structure as follows:
- HDF5 Groups -> Feature hierarchies (e.g., "observation/images/camera1")
- HDF5 Datasets -> Time-series arrays for each feature 
- HDF5 Attributes -> Metadata (timestamps, feature types, encoding info)

Key advantages over PyAV backend:
- Direct access to structured data without video container overhead
- Efficient compression for numerical data
- Native support for multi-dimensional arrays
- Parallel I/O capabilities
- Standard scientific data format
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import h5py
import numpy as np

from robodm import FeatureType
from robodm.backend.base import (
    ContainerBackend,
    Frame,
    PacketInfo,
    StreamConfig,
    StreamMetadata,
)

logger = logging.getLogger(__name__)


class HDF5Backend(ContainerBackend):
    """ContainerBackend implementation using HDF5 for structured data storage.
    
    This backend stores trajectory data in an HDF5 hierarchical format where:
    - Each feature becomes an HDF5 group/dataset
    - Time-series data is stored as HDF5 datasets with chunking and compression
    - Metadata is stored as HDF5 attributes
    - Timestamps are managed through a dedicated timestamps dataset
    
    File Structure:
    ```
    trajectory.h5
    ├── timestamps/               # Dataset: (T,) - millisecond timestamps
    ├── observation/
    │   ├── images/
    │   │   ├── camera1/         # Dataset: (T, H, W, C) - image sequence
    │   │   └── camera2/         # Dataset: (T, H, W, C) - image sequence
    │   └── state/
    │       ├── joint_positions/ # Dataset: (T, DOF) - joint angles
    │       └── gripper_state/   # Dataset: (T, 1) - gripper state
    ├── action/                  # Dataset: (T, ACTION_DIM) - actions
    └── metadata/                # Group with trajectory-level attributes
    ```
    """

    def __init__(self, compression: str = "gzip", compression_opts: int = 6):
        """Initialize HDF5Backend.
        
        Args:
            compression: HDF5 compression algorithm ("gzip", "szip", "lzf")
            compression_opts: Compression level (0-9 for gzip)
        """
        self.compression = compression
        self.compression_opts = compression_opts
        
        self.path: Optional[str] = None
        self.mode: Optional[str] = None
        self.file: Optional[h5py.File] = None
        
        # Track stream information
        self.feature_to_stream_idx: Dict[str, int] = {}
        self.stream_idx_to_feature: Dict[int, str] = {}
        self.stream_metadata: Dict[int, StreamMetadata] = {}
        
        # Buffered data for writing
        self.buffered_data: Dict[str, List[Tuple[int, Any]]] = {}  # feature -> [(timestamp, data), ...]
        self.timestamps_buffer: List[int] = []
        
        # Container compatibility attribute (for legacy Trajectory code)
        self.container: Optional[str] = None
        
    def open(self, path: str, mode: str) -> None:
        """Open HDF5 file for reading or writing."""
        if self.file is not None:
            raise RuntimeError("Backend already has an open file")
            
        if mode not in {"r", "w"}:
            raise ValueError("mode must be 'r' or 'w'")
            
        self.path = path
        self.mode = mode
        self.container = path  # For compatibility with Trajectory class
        
        try:
            if mode == "r":
                if not os.path.exists(path):
                    raise FileNotFoundError(f"HDF5 file not found: {path}")
                self.file = h5py.File(path, "r")
                self._load_stream_metadata()
            else:  # mode == "w"
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.file = h5py.File(path, "w")
                # Initialize root structure
                self._initialize_write_structure()
                
        except Exception as e:
            logger.error(f"Failed to open HDF5 file {path} in mode {mode}: {e}")
            raise
            
    def close(self) -> None:
        """Close HDF5 file and flush any pending data."""
        if self.file is None:
            return
            
        try:
            if self.mode == "w":
                # Flush any buffered data
                self._flush_buffered_data()
                
            self.file.close()
            
        except Exception as e:
            logger.error(f"Error closing HDF5 file: {e}")
        finally:
            self.file = None
            self.path = None
            self.mode = None
            self.container = None
            self.feature_to_stream_idx.clear()
            self.stream_idx_to_feature.clear()
            self.stream_metadata.clear()
            self.buffered_data.clear()
            self.timestamps_buffer.clear()
            
    def _initialize_write_structure(self) -> None:
        """Initialize HDF5 structure for writing."""
        if self.file is None:
            return
            
        # Create root metadata group
        metadata_group = self.file.create_group("metadata")
        metadata_group.attrs["robodm_version"] = "1.0"
        metadata_group.attrs["backend"] = "hdf5"
        metadata_group.attrs["created_at"] = str(np.datetime64('now'))
        
    def _load_stream_metadata(self) -> None:
        """Load stream metadata from existing HDF5 file."""
        if self.file is None:
            return
            
        stream_idx = 0
        
        def _scan_group(group_path: str, group: h5py.Group) -> None:
            nonlocal stream_idx
            
            for name, item in group.items():
                if name == "timestamps":  # Skip timestamps but not metadata
                    continue
                    
                item_path = f"{group_path}/{name}" if group_path else name
                
                if isinstance(item, h5py.Dataset):
                    # This is a feature dataset
                    feature_name = item_path
                    feature_type = item.attrs.get("feature_type", "unknown")
                    encoding = item.attrs.get("encoding", "hdf5")
                    time_base = item.attrs.get("time_base", (1, 1000))
                    
                    if isinstance(time_base, np.ndarray):
                        time_base = tuple(time_base)
                    elif not isinstance(time_base, tuple):
                        time_base = (1, 1000)
                    
                    # Register stream
                    self.feature_to_stream_idx[feature_name] = stream_idx
                    self.stream_idx_to_feature[stream_idx] = feature_name
                    self.stream_metadata[stream_idx] = StreamMetadata(
                        feature_name=feature_name,
                        feature_type=str(feature_type),
                        encoding=encoding,
                        time_base=time_base
                    )
                    stream_idx += 1
                    
                elif isinstance(item, h5py.Group):
                    # Recurse into subgroups (including metadata group)
                    _scan_group(item_path, item)
                    
        # Scan the entire file structure
        _scan_group("", self.file)
        
    def get_streams(self) -> List[StreamMetadata]:
        """Get list of all streams in the HDF5 file."""
        return [self.stream_metadata[i] for i in sorted(self.stream_metadata.keys())]
        
    def encode_data_to_packets(
        self,
        data: Any,
        stream_index: int,
        timestamp: int,
        codec_config: Any,
        force_direct_encoding: bool = False
    ) -> List[PacketInfo]:
        """Write data immediately to HDF5 instead of using packet-based approach.
        
        For HDF5, we write data directly rather than using the packet/mux paradigm.
        Returns empty list since no packets are needed.
        """
        if stream_index not in self.stream_idx_to_feature:
            raise ValueError(f"No stream with index {stream_index}")
            
        feature_name = self.stream_idx_to_feature[stream_index]
        
        # Write data immediately to HDF5
        self._write_single_timestep(feature_name, timestamp, data)
            
        return []  # No packets needed for HDF5
        
    def flush_all_streams(self) -> List[PacketInfo]:
        """Flush all buffered data to HDF5 file."""
        if self.mode == "w":
            self._flush_buffered_data()
        return []  # No packets for HDF5
        
    def _flush_buffered_data(self) -> None:
        """Write all buffered data to HDF5 datasets."""
        if self.file is None or not self.buffered_data:
            return
            
        # Sort timestamps
        unique_timestamps = sorted(set(self.timestamps_buffer))
        
        # Create or update timestamps dataset
        if "timestamps" not in self.file:
            timestamps_ds = self.file.create_dataset(
                "timestamps",
                data=np.array(unique_timestamps, dtype=np.int64),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            timestamps_ds.attrs["time_base"] = np.array([1, 1000])  # milliseconds
        else:
            # Extend existing timestamps
            existing_timestamps = self.file["timestamps"][:]
            all_timestamps = sorted(set(list(existing_timestamps) + unique_timestamps))
            del self.file["timestamps"]
            timestamps_ds = self.file.create_dataset(
                "timestamps",
                data=np.array(all_timestamps, dtype=np.int64),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            timestamps_ds.attrs["time_base"] = np.array([1, 1000])
            
        # Write feature data
        for feature_name, data_pairs in self.buffered_data.items():
            self._write_feature_data(feature_name, data_pairs, unique_timestamps)
            
        # Clear buffers
        self.buffered_data.clear()
        self.timestamps_buffer.clear()
        
    def _write_feature_data(self, feature_name: str, data_pairs: List[Tuple[int, Any]], timestamps: List[int]) -> None:
        """Write feature data to HDF5 dataset."""
        if self.file is None:
            return
            
        # Sort data by timestamp
        data_pairs = sorted(data_pairs, key=lambda x: x[0])
        
        # Align data with timestamps
        timestamp_to_data = {ts: data for ts, data in data_pairs}
        
        # Create aligned data array
        aligned_data = []
        first_data = data_pairs[0][1] if data_pairs else None
        
        if first_data is None:
            return
            
        for ts in timestamps:
            if ts in timestamp_to_data:
                aligned_data.append(timestamp_to_data[ts])
            else:
                # Fill missing timestamps with zeros or last known value
                if isinstance(first_data, np.ndarray):
                    aligned_data.append(np.zeros_like(first_data))
                else:
                    aligned_data.append(first_data)
                    
        if not aligned_data:
            return
            
        # Convert to numpy array
        try:
            data_array = np.array(aligned_data)
        except ValueError as e:
            logger.error(f"Failed to create array for feature {feature_name}: {e}")
            # Fallback to object array for heterogeneous data
            data_array = np.array(aligned_data, dtype=object)
            
        # Create HDF5 group structure
        group_path = ""
        dataset_name = feature_name
        
        if "/" in feature_name:
            parts = feature_name.split("/")
            group_path = "/".join(parts[:-1])
            dataset_name = parts[-1]
            
            # Create nested groups
            current_group = self.file
            for part in parts[:-1]:
                if part not in current_group:
                    current_group = current_group.create_group(part)
                else:
                    current_group = current_group[part]
                    
        # Create or update dataset
        full_path = feature_name
        if full_path in self.file:
            # Update existing dataset
            existing_data = self.file[full_path][:]
            combined_data = np.concatenate([existing_data, data_array], axis=0)
            del self.file[full_path]
            dataset = self.file.create_dataset(
                full_path,
                data=combined_data,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=True
            )
        else:
            # Create new dataset
            dataset = self.file.create_dataset(
                full_path,
                data=data_array,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=True
            )
            
        # Set attributes
        dataset.attrs["feature_name"] = feature_name
        if hasattr(first_data, "dtype"):
            dataset.attrs["original_dtype"] = str(first_data.dtype)
        if hasattr(first_data, "shape"):
            dataset.attrs["single_item_shape"] = first_data.shape
        dataset.attrs["encoding"] = "hdf5"
        dataset.attrs["time_base"] = np.array([1, 1000])
        
        # Set feature type
        try:
            feature_type = FeatureType.from_data(first_data)
            dataset.attrs["feature_type"] = str(feature_type)
        except Exception as e:
            logger.warning(f"Could not determine feature type for {feature_name}: {e}")
            dataset.attrs["feature_type"] = "unknown"
            
    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """Mux packet - not used in HDF5 backend (uses direct writing)."""
        pass  # HDF5 backend uses direct writing, not packet-based approach
        
    def transcode_container(
        self,
        input_path: str,
        output_path: str,
        stream_configs: Dict[int, StreamConfig],
        visualization_feature: Optional[str] = None,
    ) -> None:
        """Copy HDF5 file with potential recompression."""
        if input_path == output_path:
            return
            
        # Simple file copy for now - could implement recompression later
        import shutil
        shutil.copy2(input_path, output_path)
        
    def create_container_with_new_streams(
        self,
        original_path: str,
        new_path: str,
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig],
    ) -> Dict[int, int]:
        """Create new HDF5 file with existing and new streams and update current backend."""
        # NOTE: At this point, the backend has been closed by the trajectory's _on_new_stream method,
        # so all our internal state has been cleared. We need to rebuild everything from the original file.
        
        # Copy original file to new location (this preserves all existing data)
        import shutil
        shutil.copy2(original_path, new_path)
        
        # Reopen the new file for read/write
        if self.file is not None:
            self.file.close()
        self.path = new_path
        self.file = h5py.File(new_path, "a")  # Append mode to keep existing data
        self.mode = "w"  # Set to write mode since we'll be adding new streams
        self.container = new_path
        
        # Build new stream mappings and rebuild backend state
        stream_mapping = {}
        next_stream_idx = 0
        
        # Clear and rebuild backend state
        self.feature_to_stream_idx.clear()
        self.stream_idx_to_feature.clear()
        self.stream_metadata.clear()
        
        # First, scan existing data in the file to understand what's already there
        print(f"DEBUG: Scanning existing data in {new_path}")
        existing_features = {}
        
        def scan_datasets(name, obj):
            if isinstance(obj, h5py.Dataset) and name not in {"timestamps", "metadata"}:
                existing_features[name] = obj
                print(f"DEBUG: Found existing feature: {name} with shape {obj.shape}")
        
        self.file.visititems(scan_datasets)
        
        # Map existing streams (preserve features that are actually in the file)
        for old_idx, config in existing_streams:
            if config.feature_name in existing_features:
                stream_mapping[old_idx] = next_stream_idx
                self.feature_to_stream_idx[config.feature_name] = next_stream_idx
                self.stream_idx_to_feature[next_stream_idx] = config.feature_name
                self.stream_metadata[next_stream_idx] = StreamMetadata(
                    feature_name=config.feature_name,
                    feature_type=str(config.feature_type),
                    encoding="hdf5",
                    time_base=(1, 1000)
                )
                print(f"DEBUG: Mapped existing feature {config.feature_name} to stream {next_stream_idx}")
                next_stream_idx += 1
            else:
                print(f"DEBUG: Skipping feature {config.feature_name} - not found in file")
                
        # Add new streams
        for config in new_stream_configs:
            self.feature_to_stream_idx[config.feature_name] = next_stream_idx
            self.stream_idx_to_feature[next_stream_idx] = config.feature_name
            self.stream_metadata[next_stream_idx] = StreamMetadata(
                feature_name=config.feature_name,
                feature_type=str(config.feature_type),
                encoding="hdf5",
                time_base=(1, 1000)
            )
            print(f"DEBUG: Added new feature {config.feature_name} as stream {next_stream_idx}")
            next_stream_idx += 1
        
        print(f"DEBUG: Final stream mapping: {self.feature_to_stream_idx}")
        
        return stream_mapping
    
    def _write_single_timestep(self, feature_name: str, timestamp: int, data: Any) -> None:
        """Write a single timestep of data immediately to HDF5."""
        if self.file is None:
            return
            
        try:
            # Handle different data types
            if isinstance(data, str):
                # Convert strings to bytes for HDF5 compatibility
                data = data.encode('utf-8')
            elif not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Handle string arrays
            if isinstance(data, np.ndarray) and data.dtype.kind in {'U', 'S'}:
                # Convert Unicode or byte strings to fixed-length byte strings
                if data.dtype.kind == 'U':
                    # Unicode to bytes
                    data = data.astype('S')
                # For HDF5, we need a fixed-length string type
                if data.ndim == 0:  # Scalar string
                    max_len = len(data.item()) if hasattr(data, 'item') else len(str(data))
                    data = np.array(data, dtype=f'S{max_len}')
                else:
                    # Array of strings
                    max_len = max(len(str(item)) for item in data.flat)
                    data = data.astype(f'S{max_len}')
            
            # Create or extend dataset
            if feature_name in self.file:
                # Dataset exists, extend it
                dataset = self.file[feature_name]
                
                # Get current size
                current_size = dataset.shape[0]
                
                # Resize to accommodate new data
                new_shape = (current_size + 1,) + data.shape
                dataset.resize(new_shape)
                
                # Write new data
                dataset[current_size] = data
                
            else:
                # Create new dataset
                # Create HDF5 group structure if needed
                group_path = ""
                dataset_name = feature_name
                
                if "/" in feature_name:
                    parts = feature_name.split("/")
                    group_path = "/".join(parts[:-1])
                    dataset_name = parts[-1]
                    
                    # Create nested groups
                    current_group = self.file
                    for part in parts[:-1]:
                        if part not in current_group:
                            current_group = current_group.create_group(part)
                        else:
                            current_group = current_group[part]
                
                # Create dataset with initial data and make it extensible
                if hasattr(data, 'shape'):
                    initial_shape = (1,) + data.shape
                    max_shape = (None,) + data.shape  # Unlimited in the first dimension
                else:
                    # Handle scalar data (like bytes strings)
                    initial_shape = (1,)
                    max_shape = (None,)
                
                # Prepare data for dataset creation
                if hasattr(data, 'shape'):
                    dataset_data = np.expand_dims(data, axis=0)
                else:
                    # Handle scalar data
                    dataset_data = np.array([data])
                
                dataset = self.file.create_dataset(
                    feature_name,
                    shape=initial_shape,
                    maxshape=max_shape,
                    data=dataset_data,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    chunks=True
                )
                
                # Set attributes
                dataset.attrs["feature_name"] = feature_name
                if hasattr(data, 'dtype'):
                    dataset.attrs["original_dtype"] = str(data.dtype)
                else:
                    dataset.attrs["original_dtype"] = str(type(data))
                if hasattr(data, 'shape'):
                    dataset.attrs["single_item_shape"] = data.shape
                else:
                    dataset.attrs["single_item_shape"] = ()  # Scalar data has empty shape
                dataset.attrs["encoding"] = "hdf5"
                dataset.attrs["time_base"] = np.array([1, 1000])
                
                # Set feature type
                try:
                    feature_type = FeatureType.from_data(data)
                    dataset.attrs["feature_type"] = str(feature_type)
                except Exception as e:
                    logger.warning(f"Could not determine feature type for {feature_name}: {e}")
                    dataset.attrs["feature_type"] = "unknown"
            
            # Update or create timestamps dataset
            if "timestamps" in self.file:
                timestamps_ds = self.file["timestamps"]
                current_size = timestamps_ds.shape[0]
                timestamps_ds.resize((current_size + 1,))
                timestamps_ds[current_size] = timestamp
            else:
                timestamps_ds = self.file.create_dataset(
                    "timestamps",
                    shape=(1,),
                    maxshape=(None,),
                    data=np.array([timestamp]),
                    dtype=np.int64,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                timestamps_ds.attrs["time_base"] = np.array([1, 1000])
            
            # Force flush to disk
            self.file.flush()
                
        except Exception as e:
            logger.error(f"Error writing timestep for {feature_name}: {e}")
            import traceback
            traceback.print_exc()
        
    def validate_packet(self, packet: Any) -> bool:
        """Validate packet - always True for HDF5 since we don't use packets."""
        return True
        
    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Get iterator for reading specific streams from HDF5."""
        if self.file is None:
            raise RuntimeError("File not open")
            
        # Return a simple generator that yields data for requested streams
        def _demux_generator():
            timestamps = self.file.get("timestamps", [])
            if hasattr(timestamps, "__iter__"):
                timestamps = list(timestamps)
            else:
                timestamps = []
                
            for i, timestamp in enumerate(timestamps):
                for stream_idx in stream_indices:
                    if stream_idx in self.stream_idx_to_feature:
                        feature_name = self.stream_idx_to_feature[stream_idx]
                        if feature_name in self.file:
                            dataset = self.file[feature_name]
                            if i < len(dataset):
                                data = dataset[i]
                                
                                # Handle string decoding for byte string data
                                if isinstance(data, np.ndarray) and data.dtype.kind in ('S', 'a'):  # byte strings
                                    if data.ndim == 0:
                                        # Scalar byte string - decode to regular string
                                        data = data.item().decode('utf-8')
                                    else:
                                        # Array of byte strings - decode each element
                                        try:
                                            data = np.array([item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in data.flat]).reshape(data.shape)
                                        except (UnicodeDecodeError, AttributeError):
                                            # Keep original data if decoding fails
                                            pass
                                elif isinstance(data, bytes):
                                    # Direct bytes object - decode to string
                                    try:
                                        data = data.decode('utf-8')
                                    except UnicodeDecodeError:
                                        # Keep as bytes if decoding fails
                                        pass
                                
                                # Create a mock stream object for compatibility  
                                mock_stream = type('MockStream', (), {
                                    'index': stream_idx,
                                    'metadata': {
                                        'FEATURE_NAME': feature_name,
                                        'FEATURE_TYPE': self.stream_metadata[stream_idx].feature_type if stream_idx in self.stream_metadata else 'unknown'
                                    }
                                })()
                                
                                # Create a mock packet-like object with bytes conversion
                                class MockPacket:
                                    def __init__(self):
                                        self.pts = timestamp
                                        self.dts = timestamp
                                        self.data = data
                                        self.stream_index = stream_idx
                                        self.feature_name = feature_name
                                        self.stream = mock_stream
                                    
                                    def __bytes__(self):
                                        # Return pickled data for decode_stream_frames compatibility
                                        import pickle
                                        return pickle.dumps(self.data)
                                
                                packet = MockPacket()
                                yield packet
                                
        return _demux_generator()
        
    def seek_container(self, timestamp: int, stream_index: int, any_frame: bool = True) -> None:
        """Seek to specific timestamp - HDF5 allows random access."""
        # HDF5 naturally supports random access, so seeking is essentially a no-op
        # In a more sophisticated implementation, we could maintain current position state
        pass
        
    def decode_stream_frames(self, stream_index: int, packet_data: Optional[bytes] = None) -> List[Any]:
        """Decode frames from HDF5 stream."""
        if self.file is None:
            raise RuntimeError("File not open")
            
        if stream_index not in self.stream_idx_to_feature:
            raise ValueError(f"No stream with index {stream_index}")
            
        feature_name = self.stream_idx_to_feature[stream_index]
        
        if packet_data is None:
            # Return all data for this feature
            if feature_name in self.file:
                dataset = self.file[feature_name]
                return [dataset[i] for i in range(len(dataset))]
            else:
                return []
        else:
            # Decode specific packet data (not typically used for HDF5)
            return [pickle.loads(packet_data) if isinstance(packet_data, bytes) else packet_data]
            
    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get codec name for stream."""
        if stream_index in self.stream_metadata:
            return self.stream_metadata[stream_index].encoding
        return "hdf5"
        
    def convert_frame_to_array(self, frame: Any, feature_type: Any, format: str = "rgb24") -> Any:
        """Convert frame to array - HDF5 stores arrays directly."""
        # HDF5 backend stores numpy arrays directly, so conversion is minimal
        if isinstance(frame, np.ndarray):
            # Handle string data - decode bytes to strings if it's string feature type
            if hasattr(feature_type, 'dtype_info') and 'string' in str(feature_type):
                if frame.dtype.kind in ('S', 'a'):  # bytes or byte strings
                    # Convert bytes array to string
                    if frame.ndim == 0:
                        # Scalar bytes
                        return frame.item().decode('utf-8')
                    else:
                        # Array of bytes - decode each element
                        return np.array([item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in frame.flat]).reshape(frame.shape)
            return frame
        elif hasattr(frame, 'data'):
            # Handle mock packet objects from demux_streams - recursively process the data
            return self.convert_frame_to_array(frame.data, feature_type, format)
        elif isinstance(frame, bytes):
            # Handle pickled data or direct bytes
            try:
                return pickle.loads(frame)
            except:
                # If not pickled, try to decode as utf-8
                return frame.decode('utf-8')
        else:
            return frame
            
    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if stream exists for feature name."""
        return self.feature_to_stream_idx.get(feature_name)
        
    # Additional HDF5-specific helper methods
    
    def add_stream_for_feature(
        self, 
        feature_name: str, 
        feature_type: "FeatureType", 
        codec_config: Any,
        encoding: Optional[str] = None
    ) -> int:
        """Add a new stream for a feature (HDF5-specific helper).
        
        Args:
            feature_name: Name of the feature
            feature_type: FeatureType object describing the data
            codec_config: Codec configuration (not used for HDF5 but kept for compatibility)
            encoding: Optional encoding specification (not used for HDF5)
            
        Returns:
            Stream index for the newly created stream
        """
        if feature_name in self.feature_to_stream_idx:
            return self.feature_to_stream_idx[feature_name]
            
        # Find next available stream index
        next_idx = max(self.stream_idx_to_feature.keys()) + 1 if self.stream_idx_to_feature else 0
        
        # Register stream
        self.feature_to_stream_idx[feature_name] = next_idx
        self.stream_idx_to_feature[next_idx] = feature_name
        self.stream_metadata[next_idx] = StreamMetadata(
            feature_name=feature_name,
            feature_type=str(feature_type),
            encoding="hdf5",
            time_base=(1, 1000)
        )
        
        return next_idx
        
    def read_feature_data(self, feature_name: str, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """Read data for a specific feature from HDF5."""
        if self.file is None or feature_name not in self.file:
            return None
            
        dataset = self.file[feature_name]
        
        if start_idx is None and end_idx is None:
            return dataset[:]
        elif end_idx is None:
            return dataset[start_idx:]
        elif start_idx is None:
            return dataset[:end_idx]
        else:
            return dataset[start_idx:end_idx]
            
    def get_timestamps(self) -> Optional[np.ndarray]:
        """Get timestamps array from HDF5."""
        if self.file is None or "timestamps" not in self.file:
            return None
        return self.file["timestamps"][:]
        
    def get_trajectory_length(self) -> int:
        """Get number of timesteps in trajectory."""
        if self.file is None:
            return 0
        if "timestamps" in self.file:
            return len(self.file["timestamps"])
        # Fallback: find the first dataset and use its length
        for item in self.file.values():
            if isinstance(item, h5py.Dataset):
                return len(item)
        return 0