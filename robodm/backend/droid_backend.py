"""DROID-backed implementation of the ContainerBackend interface.

This module provides a native DROID storage backend for RoboDM trajectories,
offering direct access to DROID raw format without intermediate conversion.

The DROID backend maps DROID concepts to RoboDM structure as follows:
- DROID trajectory.h5 -> Numerical data (actions, observations, robot state)
- DROID recordings/MP4/*.mp4 -> Video streams for each camera
- DROID metadata_*.json -> Camera mappings and trajectory metadata

Key advantages over HDF5 backend with conversion:
- Direct access to DROID raw format without conversion overhead
- Native support for DROID camera naming conventions
- Preserves original DROID data structure and metadata
- Eliminates intermediate file creation
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
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


class DROIDBackend(ContainerBackend):
    """ContainerBackend implementation for DROID raw trajectory directories.
    
    This backend loads trajectory data directly from DROID raw format:
    - trajectory.h5: Contains actions, observations, robot state, timestamps
    - recordings/MP4/*.mp4: Video files for each camera
    - metadata_*.json: Camera mappings and trajectory metadata
    
    Directory Structure:
    ```
    trajectory_directory/
    ├── metadata_[lab]+[uuid]+[timestamp].json  # Metadata
    ├── trajectory.h5                           # HDF5 with numerical data
    └── recordings/
        ├── MP4/                                # MP4 video files  
        │   ├── [camera_serial].mp4
        │   └── ...
        └── SVO/                                # SVO files (optional)
            ├── [camera_serial].svo
            └── ...
    ```
    
    Feature Mapping to RoboDM:
    - action/joint_position -> action
    - observation/robot_state/* -> observation/state/*
    - MP4 files -> observation/images/[camera_name]
    - metadata -> metadata/*
    """

    def __init__(self, video_path_key: Optional[str] = None):
        """Initialize DROID Backend.
        
        Args:
            video_path_key: Specific video path key from metadata (e.g., 'ext1_mp4_path', 'wrist_mp4_path')
        """
        self.path: Optional[str] = None
        self.mode: Optional[str] = None
        self.video_path_key = video_path_key
        
        # DROID data files
        self.trajectory_h5: Optional[h5py.File] = None
        self.metadata: Optional[Dict] = None
        self.video_files: Dict[str, str] = {}  # camera_serial -> mp4_path
        
        # Track stream information
        self.feature_to_stream_idx: Dict[str, int] = {}
        self.stream_idx_to_feature: Dict[int, str] = {}
        self.stream_metadata: Dict[int, StreamMetadata] = {}
        
        # Video capture objects (cached)
        self._video_caps: Dict[str, cv2.VideoCapture] = {}
        
        # Container compatibility
        self.container: Optional[str] = None
        
    def open(self, path: str, mode: str) -> None:
        """Open DROID trajectory directory for reading."""
        if self.trajectory_h5 is not None:
            raise RuntimeError("Backend already has an open trajectory")
            
        if mode not in {"r"}:  # Only read mode supported for DROID
            raise ValueError("DROID backend only supports read mode 'r'")
            
        if not os.path.isdir(path):
            raise FileNotFoundError(f"DROID trajectory directory not found: {path}")
            
        self.path = path
        self.mode = mode
        self.container = path
        
        try:
            self._load_droid_data()
            self._setup_streams()
        except Exception as e:
            logger.error(f"Failed to open DROID trajectory {path}: {e}")
            raise
            
    def close(self) -> None:
        """Close DROID trajectory and cleanup resources."""
        if self.trajectory_h5 is not None:
            self.trajectory_h5.close()
            
        # Close video capture objects
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps.clear()
            
        # Reset state
        self.trajectory_h5 = None
        self.metadata = None
        self.video_files.clear()
        self.path = None
        self.mode = None
        self.container = None
        self.feature_to_stream_idx.clear()
        self.stream_idx_to_feature.clear()
        self.stream_metadata.clear()
        
    def _load_droid_data(self) -> None:
        """Load DROID trajectory data from directory."""
        if self.path is None:
            return
            
        # Load trajectory.h5
        h5_path = os.path.join(self.path, "trajectory.h5")
        if os.path.exists(h5_path):
            self.trajectory_h5 = h5py.File(h5_path, "r")
        else:
            raise FileNotFoundError(f"trajectory.h5 not found in {self.path}")
            
        # Load metadata JSON
        metadata_files = list(Path(self.path).glob("metadata_*.json"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.warning(f"No metadata JSON found in {self.path}")
            self.metadata = {}
            
        # Find MP4 video files
        mp4_dir = os.path.join(self.path, "recordings", "MP4")
        if os.path.exists(mp4_dir):
            if self.video_path_key and self.metadata and self.video_path_key in self.metadata:
                # Use specific video path from metadata
                relative_path = self.metadata[self.video_path_key]
                video_filename = os.path.basename(relative_path)
                local_video_path = os.path.join(mp4_dir, video_filename)
                
                if os.path.exists(local_video_path):
                    camera_serial = video_filename.replace('.mp4', '')
                    self.video_files[camera_serial] = local_video_path
                    logger.info(f"Using specified video: {self.video_path_key} -> {video_filename}")
                else:
                    logger.warning(f"Specified video {self.video_path_key} not found: {local_video_path}")
            
            if not self.video_files:
                # Fallback: load all MP4 files
                for mp4_file in os.listdir(mp4_dir):
                    if mp4_file.endswith('.mp4'):
                        camera_serial = mp4_file.replace('.mp4', '')
                        self.video_files[camera_serial] = os.path.join(mp4_dir, mp4_file)
        
        logger.info(f"Loaded DROID trajectory with {len(self.video_files)} video files")
        
    def _setup_streams(self) -> None:
        """Setup stream metadata from DROID data."""
        stream_idx = 0
        
        # Add streams for HDF5 numerical data
        if self.trajectory_h5 is not None:
            # Actions
            if "action" in self.trajectory_h5:
                action_group = self.trajectory_h5["action"]
                if "joint_position" in action_group:
                    feature_name = "action"
                    self.feature_to_stream_idx[feature_name] = stream_idx
                    self.stream_idx_to_feature[stream_idx] = feature_name
                    self.stream_metadata[stream_idx] = StreamMetadata(
                        feature_name=feature_name,
                        feature_type=str(FeatureType(dtype="float32", shape=(8,))),
                        encoding="droid_h5",
                        time_base=(1, 1000)
                    )
                    stream_idx += 1
            
            # Observations - robot state
            if "observation" in self.trajectory_h5 and "robot_state" in self.trajectory_h5["observation"]:
                robot_state = self.trajectory_h5["observation"]["robot_state"]
                for key in robot_state.keys():
                    feature_name = f"observation/state/{key}"
                    self.feature_to_stream_idx[feature_name] = stream_idx
                    self.stream_idx_to_feature[stream_idx] = feature_name
                    self.stream_metadata[stream_idx] = StreamMetadata(
                        feature_name=feature_name,
                        feature_type=str(FeatureType(dtype="float32", shape=(-1,))),
                        encoding="droid_h5",
                        time_base=(1, 1000)
                    )
                    stream_idx += 1
        
        # Add streams for video data
        camera_mapping = self._get_camera_mapping()
        for camera_serial, mp4_path in self.video_files.items():
            camera_name = camera_mapping.get(camera_serial, f"camera_{camera_serial}")
            feature_name = f"observation/images/{camera_name}"
            
            self.feature_to_stream_idx[feature_name] = stream_idx
            self.stream_idx_to_feature[stream_idx] = feature_name
            self.stream_metadata[stream_idx] = StreamMetadata(
                feature_name=feature_name,
                feature_type=str(FeatureType(dtype="uint8", shape=(720,1280,3))),
                encoding="mp4",
                time_base=(1, 30)  # Assume 30 FPS for MP4
            )
            stream_idx += 1
            
        # Add metadata stream
        if self.metadata:
            feature_name = "metadata/language_instruction"
            self.feature_to_stream_idx[feature_name] = stream_idx
            self.stream_idx_to_feature[stream_idx] = feature_name
            self.stream_metadata[stream_idx] = StreamMetadata(
                feature_name=feature_name,
                feature_type=str(FeatureType(dtype="str", shape=())),
                encoding="json",
                time_base=(1, 1000)
            )
            stream_idx += 1
            
    def _get_camera_mapping(self) -> Dict[str, str]:
        """Get mapping from camera serial to camera name."""
        if not self.metadata:
            return {}
            
        mapping = {}
        # Map based on metadata camera information
        if "wrist_cam_serial" in self.metadata:
            mapping[self.metadata["wrist_cam_serial"]] = "exterior_image_1_left"  # Match droid_hdf5_pipeline expectation
        if "ext1_cam_serial" in self.metadata:
            mapping[self.metadata["ext1_cam_serial"]] = "exterior_image_2_left"
        if "ext2_cam_serial" in self.metadata:
            mapping[self.metadata["ext2_cam_serial"]] = "exterior_image_3_left"
            
        return mapping
        
    def get_streams(self) -> List[StreamMetadata]:
        """Get list of all streams in the DROID trajectory."""
        return [self.stream_metadata[i] for i in sorted(self.stream_metadata.keys())]
        
    def encode_data_to_packets(
        self,
        data: Any,
        stream_index: int,
        timestamp: int,
        codec_config: Any,
        force_direct_encoding: bool = False
    ) -> List[PacketInfo]:
        """DROID backend is read-only."""
        raise NotImplementedError("DROID backend is read-only")
        
    def flush_all_streams(self) -> List[PacketInfo]:
        """DROID backend is read-only."""
        return []
        
    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """DROID backend is read-only."""
        raise NotImplementedError("DROID backend is read-only")
        
    def transcode_container(
        self,
        input_path: str,
        output_path: str,
        stream_configs: Dict[int, StreamConfig],
        visualization_feature: Optional[str] = None,
    ) -> None:
        """DROID backend is read-only."""
        raise NotImplementedError("DROID backend is read-only")
        
    def create_container_with_new_streams(
        self,
        original_path: str,
        new_path: str,
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig],
    ) -> Dict[int, int]:
        """DROID backend is read-only."""
        raise NotImplementedError("DROID backend is read-only")
    
    def validate_packet(self, packet: Any) -> bool:
        """Validate packet - always True for DROID since we generate them."""
        return True
        
    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Get iterator for reading specific streams from DROID data."""
        if self.trajectory_h5 is None:
            raise RuntimeError("Trajectory not open")
            
        def _demux_generator():
            # Determine trajectory length
            traj_length = self.get_trajectory_length()
            
            for timestep in range(traj_length):
                timestamp = self._get_timestamp(timestep)
                
                for stream_idx in stream_indices:
                    if stream_idx in self.stream_idx_to_feature:
                        feature_name = self.stream_idx_to_feature[stream_idx]
                        data = self._get_feature_data(feature_name, timestep)
                        
                        if data is not None:
                            # Create mock packet
                            class MockPacket:
                                def __init__(self, stream_idx, feature_name, timestamp, data, backend_ref):
                                    self.pts = timestamp
                                    self.dts = timestamp
                                    self.data = data
                                    self.stream_index = stream_idx
                                    self.feature_name = feature_name
                                    
                                    # Mock stream object
                                    self.stream = type('MockStream', (), {
                                        'index': stream_idx,
                                        'metadata': {
                                            'FEATURE_NAME': feature_name,
                                            'FEATURE_TYPE': backend_ref._get_feature_type(feature_name)
                                        }
                                    })()
                                
                                def __bytes__(self):
                                    return pickle.dumps(self.data)
                            
                            packet = MockPacket(stream_idx, feature_name, timestamp, data, self)
                            yield packet
                                
        return _demux_generator()
    
    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory in timesteps."""
        if self.trajectory_h5 is None:
            return 0
            
        # Use action data to determine length
        if "action" in self.trajectory_h5 and "joint_position" in self.trajectory_h5["action"]:
            return len(self.trajectory_h5["action"]["joint_position"])
            
        # Fallback: use robot state
        if ("observation" in self.trajectory_h5 and 
            "robot_state" in self.trajectory_h5["observation"] and
            "joint_positions" in self.trajectory_h5["observation"]["robot_state"]):
            return len(self.trajectory_h5["observation"]["robot_state"]["joint_positions"])
            
        return 0
        
    def _get_timestamp(self, timestep: int) -> int:
        """Get timestamp for a given timestep."""
        if (self.trajectory_h5 is not None and
            "observation" in self.trajectory_h5 and
            "timestamp" in self.trajectory_h5["observation"] and
            "control" in self.trajectory_h5["observation"]["timestamp"] and
            "step_start" in self.trajectory_h5["observation"]["timestamp"]["control"]):
            timestamps = self.trajectory_h5["observation"]["timestamp"]["control"]["step_start"]
            if timestep < len(timestamps):
                # Convert nanoseconds to milliseconds
                return int(timestamps[timestep] / 1000000)
        
        # Fallback: use timestep index as milliseconds
        return timestep * 33  # Assume ~30 FPS
        
    def _get_feature_data(self, feature_name: str, timestep: int) -> Any:
        """Get data for a specific feature at a timestep."""
        if feature_name == "action":
            return self._get_action_data(timestep)
        elif feature_name.startswith("observation/state/"):
            state_key = feature_name.replace("observation/state/", "")
            return self._get_observation_data(state_key, timestep)
        elif feature_name.startswith("observation/images/"):
            camera_name = feature_name.replace("observation/images/", "")
            return self._get_image_data(camera_name, timestep)
        elif feature_name == "metadata/language_instruction":
            return self._get_language_instruction()
        else:
            logger.warning(f"Unknown feature: {feature_name}")
            return None
            
    def _get_action_data(self, timestep: int) -> Optional[np.ndarray]:
        """Get action data for a timestep."""
        if (self.trajectory_h5 is None or 
            "action" not in self.trajectory_h5 or
            "joint_position" not in self.trajectory_h5["action"]):
            return None
            
        action_group = self.trajectory_h5["action"]
        
        # Combine action components
        components = []
        if "joint_position" in action_group and timestep < len(action_group["joint_position"]):
            components.append(action_group["joint_position"][timestep])
        if "gripper_position" in action_group and timestep < len(action_group["gripper_position"]):
            components.append([action_group["gripper_position"][timestep]])
            
        if components:
            return np.concatenate(components).astype(np.float32)
        return None
        
    def _get_observation_data(self, state_key: str, timestep: int) -> Optional[np.ndarray]:
        """Get observation data for a timestep."""
        if (self.trajectory_h5 is None or 
            "observation" not in self.trajectory_h5 or
            "robot_state" not in self.trajectory_h5["observation"]):
            return None
            
        robot_state = self.trajectory_h5["observation"]["robot_state"]
        if state_key in robot_state and timestep < len(robot_state[state_key]):
            return np.array(robot_state[state_key][timestep]).astype(np.float32)
        return None
        
    def _get_image_data(self, camera_name: str, timestep: int) -> Optional[np.ndarray]:
        """Get image data for a camera at a timestep."""
        # Find the camera serial for this camera name
        camera_mapping = self._get_camera_mapping()
        camera_serial = None
        for serial, name in camera_mapping.items():
            if name == camera_name:
                camera_serial = serial
                break
                
        if camera_serial is None or camera_serial not in self.video_files:
            return None
            
        # Get video capture object (cached)
        if camera_serial not in self._video_caps:
            mp4_path = self.video_files[camera_serial]
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {mp4_path}")
                return None
            self._video_caps[camera_serial] = cap
        
        cap = self._video_caps[camera_serial]
        
        # Seek to the right frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, timestep)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
        
    def _get_language_instruction(self) -> Optional[str]:
        """Get language instruction from metadata."""
        if self.metadata and "current_task" in self.metadata:
            return self.metadata["current_task"]
        return None
        
    def _get_feature_type(self, feature_name: str) -> str:
        """Get feature type for a feature name."""
        if stream_idx := self.feature_to_stream_idx.get(feature_name):
            if stream_idx in self.stream_metadata:
                return self.stream_metadata[stream_idx].feature_type
        return "unknown"
        
    def seek_container(self, timestamp: int, stream_index: int, any_frame: bool = True) -> None:
        """Seek to specific timestamp."""
        # DROID allows random access, so seeking is essentially a no-op
        pass
        
    def decode_stream_frames(self, stream_index: int, packet_data: Optional[bytes] = None) -> List[Any]:
        """Decode frames from DROID stream."""
        if packet_data is None:
            return []
        else:
            return [pickle.loads(packet_data) if isinstance(packet_data, bytes) else packet_data]
            
    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get codec name for stream."""
        if stream_index in self.stream_metadata:
            return self.stream_metadata[stream_index].encoding
        return "droid"
        
    def convert_frame_to_array(self, frame: Any, feature_type: Any, format: str = "rgb24") -> Any:
        """Convert frame to array."""
        if isinstance(frame, np.ndarray):
            return frame
        elif hasattr(frame, 'data'):
            return frame.data
        elif isinstance(frame, bytes):
            try:
                return pickle.loads(frame)
            except:
                return frame.decode('utf-8') if isinstance(frame, bytes) else frame
        else:
            return frame
            
    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if stream exists for feature name."""
        return self.feature_to_stream_idx.get(feature_name)