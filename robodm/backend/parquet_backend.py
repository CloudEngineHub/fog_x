import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from robodm import FeatureType
from robodm.backend.base import (
    ContainerBackend,
    Frame,
    PacketInfo,
    StreamConfig,
    StreamMetadata,
)


class ParquetBackend(ContainerBackend):
    """
    Parquet backend that bypasses encoding/decoding overhead.
    
    Assumes data is already aligned with format: (timestamp, feature_data_1, feature_data_2, ...)
    This backend directly writes structured data to parquet without video container overhead.
    """

    def __init__(self):
        self.path: Optional[str] = None
        self.mode: Optional[str] = None
        self.data_rows: List[Dict[str, Any]] = []
        self.feature_types: Dict[str, FeatureType] = {}
        self.feature_columns: List[str] = []
        self._is_open = False
        self.container: Optional[str] = None  # For compatibility with Trajectory class

    def open(self, path: str, mode: str) -> None:
        """Open a parquet file for reading or writing"""
        if self._is_open:
            raise RuntimeError("Backend is already open")
            
        self.path = path
        self.mode = mode
        self._is_open = True
        self.container = path  # Set container to path for compatibility
        
        if mode == "r":
            if not os.path.exists(path):
                raise FileNotFoundError(f"Parquet file not found: {path}")
            self._load_metadata()
        elif mode == "w":
            self.data_rows = []
            self.feature_types = {}
            self.feature_columns = []
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'r' or 'w'")

    def close(self) -> None:
        """Close the parquet file and write data if in write mode"""
        if not self._is_open:
            return
            
        if self.mode == "w" and self.data_rows:
            self._write_to_parquet()
            
        self._is_open = False
        self.path = None
        self.mode = None
        self.container = None

    def _load_metadata(self) -> None:
        """Load metadata from existing parquet file"""
        if not self.path or not os.path.exists(self.path):
            return
            
        try:
            parquet_file = pq.ParquetFile(self.path)
            schema_metadata = parquet_file.metadata.metadata
            
            if schema_metadata and b'robodm_features' in schema_metadata:
                features_metadata = pickle.loads(schema_metadata[b'robodm_features'])
                for feature_name, feature_type_str in features_metadata.items():
                    self.feature_types[feature_name] = FeatureType.from_str(feature_type_str)
                    
            # Get column names (excluding timestamp)
            schema = parquet_file.schema.to_arrow_schema()
            self.feature_columns = [name for name in schema.names if name != 'timestamp']
            
        except Exception as e:
            warnings.warn(f"Could not load parquet metadata: {e}")

    def _write_to_parquet(self) -> None:
        """Write aligned data rows to parquet file"""
        if not self.data_rows or not self.path:
            return
            
        # Convert to DataFrame with aligned structure
        df = pd.DataFrame(self.data_rows)
        
        # Serialize complex data types
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            # Handle numpy arrays and complex objects
            if df[col].dtype == object:
                first_val = df[col].iloc[0]
                if isinstance(first_val, np.ndarray):
                    # Store arrays as bytes
                    df[col] = df[col].apply(lambda x: x.tobytes() if isinstance(x, np.ndarray) else pickle.dumps(x))
                else:
                    # Pickle other objects
                    df[col] = df[col].apply(pickle.dumps)
        
        # Create Arrow table with metadata
        table = pa.Table.from_pandas(df)
        
        # Add feature type metadata
        features_metadata = {name: str(ftype) for name, ftype in self.feature_types.items()}
        existing_metadata = table.schema.metadata or {}
        existing_metadata[b'robodm_features'] = pickle.dumps(features_metadata)
        table = table.replace_schema_metadata(existing_metadata)
        
        # Write to parquet
        pq.write_table(table, self.path, compression='snappy')

    def add_aligned_row(self, timestamp: int, feature_data: Dict[str, Any]) -> None:
        """Add a row of aligned data (timestamp + all features for that timestamp)"""
        row = {'timestamp': timestamp}
        row.update(feature_data)
        
        # Track feature types from first occurrence
        for feature_name, data in feature_data.items():
            if feature_name not in self.feature_types:
                self.feature_types[feature_name] = FeatureType.from_data(data)
                if feature_name not in self.feature_columns:
                    self.feature_columns.append(feature_name)
                    
        self.data_rows.append(row)

    def get_streams(self) -> List[StreamMetadata]:
        """Get list of all streams (features) in the parquet file"""
        streams = []
        
        for i, feature_name in enumerate(self.feature_columns):
            feature_type = self.feature_types.get(feature_name)
            metadata = StreamMetadata(
                feature_name=feature_name,
                feature_type=str(feature_type) if feature_type else "unknown",
                encoding="parquet",
                time_base=(1, 1000),  # milliseconds
            )
            streams.append(metadata)
            
        return streams

    def encode_data_to_packets(self, data: Any, stream_index: int,
                              timestamp: int, codec_config: Any) -> List[PacketInfo]:
        """Buffer data for aligned writing - returns empty list since no packets needed"""
        if stream_index >= len(self.feature_columns):
            raise ValueError(f"Stream index {stream_index} out of range")
            
        feature_name = self.feature_columns[stream_index]
        
        # Find or create row for this timestamp
        row = None
        for existing_row in self.data_rows:
            if existing_row['timestamp'] == timestamp:
                row = existing_row
                break
                
        if row is None:
            row = {'timestamp': timestamp}
            self.data_rows.append(row)
            
        row[feature_name] = data
        
        # Track feature type
        if feature_name not in self.feature_types:
            self.feature_types[feature_name] = FeatureType.from_data(data)
            
        return []

    def flush_all_streams(self) -> List[PacketInfo]:
        """Flush all streams - no-op for parquet backend"""
        return []

    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """Mux packet - no-op for parquet backend"""
        pass

    def transcode_container(self, input_path: str, output_path: str,
                          stream_configs: Dict[int, StreamConfig],
                          visualization_feature: Optional[str] = None) -> None:
        """Transcode container - copy for parquet backend"""
        if input_path != output_path:
            import shutil
            shutil.copy(input_path, output_path)

    def create_container_with_new_streams(
        self, original_path: str, new_path: str,
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig]
    ) -> Dict[int, int]:
        """Create new container with additional streams"""
        # Copy existing file
        import shutil
        shutil.copy(original_path, new_path)
        
        # Update feature types with new streams
        current_index = len(self.feature_columns)
        stream_mapping = {}
        
        for old_index, config in existing_streams:
            stream_mapping[old_index] = old_index
            
        for config in new_stream_configs:
            self.feature_types[config.feature_name] = config.feature_type
            self.feature_columns.append(config.feature_name)
            stream_mapping[len(stream_mapping)] = current_index
            current_index += 1
            
        return stream_mapping

    def validate_packet(self, packet: Any) -> bool:
        """Validate packet - always true for parquet backend"""
        return True

    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Demux streams from parquet file"""
        if self.mode != "r" or not self.path:
            return iter([])
            
        try:
            df = pd.read_parquet(self.path)
            packets = []
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                
                for stream_idx in stream_indices:
                    if stream_idx >= len(self.feature_columns):
                        continue
                        
                    feature_name = self.feature_columns[stream_idx]
                    if feature_name not in row:
                        continue
                        
                    data = row[feature_name]
                    
                    # Create mock packet
                    packet = type('MockPacket', (), {
                        'stream': type('MockStream', (), {'index': stream_idx})(),
                        'pts': timestamp,
                        'data': data
                    })()
                    packets.append(packet)
                    
            return iter(packets)
            
        except Exception:
            return iter([])

    def seek_container(self, timestamp: int, stream_index: int,
                      any_frame: bool = True) -> None:
        """Seek container - no-op for parquet backend"""
        pass

    def decode_stream_frames(self, stream_index: int,
                           packet_data: Optional[bytes] = None) -> List[Any]:
        """Decode frames from parquet data"""
        if packet_data is None:
            return []
            
        if stream_index >= len(self.feature_columns):
            return []
            
        feature_name = self.feature_columns[stream_index]
        feature_type = self.feature_types.get(feature_name)
        
        # Decode based on feature type
        if isinstance(packet_data, bytes):
            try:
                # Try to deserialize as numpy array first
                if feature_type and hasattr(feature_type, 'shape') and feature_type.shape:
                    arr = np.frombuffer(packet_data, dtype=feature_type.dtype)
                    arr = arr.reshape(feature_type.shape)
                    return [arr]
                else:
                    # Try pickle
                    return [pickle.loads(packet_data)]
            except Exception:
                return [packet_data]
        else:
            return [packet_data]

    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get codec name for stream"""
        return "parquet"

    def convert_frame_to_array(self, frame: Any, feature_type: Any,
                             format: str = "rgb24") -> Any:
        """Convert frame to array - direct pass-through for parquet"""
        return frame

    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if stream exists for feature"""
        try:
            return self.feature_columns.index(feature_name)
        except ValueError:
            return None

    def add_stream_for_feature(self, feature_name: str, feature_type: FeatureType,
                             codec_config: Any, encoding: str) -> None:
        """Add stream for feature"""
        if feature_name not in self.feature_types:
            self.feature_types[feature_name] = feature_type
            self.feature_columns.append(feature_name)

    def create_streams_for_batch_data(self, sample_data: Dict[str, Any], codec_config: Any,
                                    feature_name_separator: str = "/",
                                    visualization_feature: Optional[str] = None) -> Dict[str, int]:
        """Create streams for batch data processing - compatibility method for parquet backend"""
        from robodm.utils.flatten import _flatten_dict
        
        # Flatten the sample data to get all feature names
        flattened_sample = _flatten_dict(sample_data, sep=feature_name_separator)
        
        feature_to_stream_idx = {}
        for i, (feature_name, sample_value) in enumerate(flattened_sample.items()):
            feature_type = FeatureType.from_data(sample_value)
            self.feature_types[feature_name] = feature_type
            if feature_name not in self.feature_columns:
                self.feature_columns.append(feature_name)
            feature_to_stream_idx[feature_name] = i
            
        return feature_to_stream_idx

    def encode_batch_data_directly(self, data_batch: List[Dict[str, Any]], 
                                 feature_to_stream_idx: Dict[str, int],
                                 codec_config: Any, feature_name_separator: str = "/",
                                 fps: Optional[Union[int, Dict[str, int]]] = None) -> None:
        """Encode batch data directly - compatibility method for parquet backend"""
        from robodm.utils.flatten import _flatten_dict
        
        # Convert batch data to aligned format
        for i, step_dict in enumerate(data_batch):
            timestamp_ms = i * 100  # Default 100ms intervals, could be made configurable
            
            # Flatten the step data
            flattened_step = _flatten_dict(step_dict, sep=feature_name_separator)
            
            row_data = {"timestamp": timestamp_ms}
            row_data.update(flattened_step)
            
            self.data_rows.append(row_data)