"""
DROID Dataset integration with RoboDM Agent system.

This module provides a dataset interface for DROID trajectories that works
with the natural language Agent interface, enabling operations like:
    agent = Agent(droid_dataset)
    agent.filter("trajectories that are successful")
"""

import glob
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import ray.data as rd
from ray.data import Dataset

from robodm.backend.droid_backend import DROIDBackend
from robodm.dataset import DatasetConfig

logger = logging.getLogger(__name__)


def load_droid_trajectory_simple(trajectory_path: str) -> Dict[str, Any]:
    """
    Load a DROID trajectory into a simple dictionary format.
    
    This provides a simplified interface for loading trajectory data
    that works well with the Agent system.
    """
    try:
        backend = DROIDBackend()
        backend.open(trajectory_path, "r")
        
        # Extract basic trajectory data
        trajectory_length = backend.get_trajectory_length()
        
        data = {
            "trajectory_length": trajectory_length,
            "features": {}
        }
        
        # Get available streams
        streams = backend.get_streams()
        stream_names = [stream.feature_name for stream in streams]
        
        # Load key data
        for stream in streams:
            feature_name = stream.feature_name
            
            # Only load first/last frames for images to save memory
            if "images" in feature_name:
                # Load first and last frame only
                first_frame = backend._get_feature_data(feature_name, 0)
                last_frame = backend._get_feature_data(feature_name, trajectory_length - 1)
                data["features"][feature_name] = {
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "shape": first_frame.shape if first_frame is not None else None
                }
            elif "metadata" in feature_name:
                # Load metadata
                metadata_val = backend._get_feature_data(feature_name, 0)
                data["features"][feature_name] = metadata_val
            else:
                # Load small numerical data completely
                values = []
                for timestep in range(min(trajectory_length, 10)):  # Sample first 10 steps
                    val = backend._get_feature_data(feature_name, timestep)
                    if val is not None:
                        values.append(val)
                data["features"][feature_name] = values
        
        backend.close()
        return data
        
    except Exception as e:
        logger.error(f"Failed to load DROID trajectory {trajectory_path}: {e}")
        return {"error": str(e)}


@dataclass 
class DroidDatasetConfig(DatasetConfig):
    """Configuration for DroidDataset."""
    
    auto_download: bool = True
    download_workers: int = 4
    temp_dir: Optional[str] = None
    

class DroidDataset:
    """
    DROID Dataset with Agent system integration.
    
    Provides a Ray Dataset interface for DROID trajectories that works
    with the Agent system for natural language processing.
    
    Features:
    - Lazy loading and downloading of DROID trajectories
    - Ray Dataset interface compatible with Agent system
    - Integration with existing DROID backend
    - Parallel processing and filtering capabilities
    """
    
    def __init__(
        self,
        trajectory_paths: Union[str, List[str]], 
        local_dir: Optional[str] = None,
        config: Optional[DroidDatasetConfig] = None,
        **kwargs
    ):
        """
        Initialize DROID dataset.
        
        Args:
            trajectory_paths: Either GCS paths or local paths to DROID trajectories
            local_dir: Directory for downloaded trajectories (if downloading)
            config: Dataset configuration
            **kwargs: Additional arguments
        """
        if not ray.is_initialized():
            ray.init()
            
        self.config = config or DroidDatasetConfig()
        
        # Handle trajectory paths
        if isinstance(trajectory_paths, str):
            if "*" in trajectory_paths or trajectory_paths.startswith("gs://"):
                # Pattern or GCS path - scan for trajectories
                self.trajectory_paths = self._scan_trajectories(trajectory_paths)
            elif os.path.isdir(trajectory_paths):
                # Local directory - find DROID trajectories
                self.trajectory_paths = self._find_local_trajectories(trajectory_paths)
            else:
                # Single path
                self.trajectory_paths = [trajectory_paths]
        else:
            self.trajectory_paths = trajectory_paths
            
        self.local_dir = local_dir or tempfile.mkdtemp(prefix="droid_dataset_")
        
        # Track download state
        self._downloaded_paths = {}  # maps gcs_path -> local_path
        self._is_downloaded = False
        
        # Create Ray dataset from trajectory paths with metadata
        self.ray_dataset = self._create_initial_dataset()
        
        logger.info(f"Initialized DroidDataset with {len(self.trajectory_paths)} trajectories")
        
    def _scan_trajectories(self, pattern_or_path: str) -> List[str]:
        """Scan for DROID trajectories from pattern or GCS path."""
        if pattern_or_path.startswith("gs://"):
            # Use GCS scanning from the pipeline
            from examples.droid_h5.droid_hdf5_pipeline import scan_droid_trajectories
            return scan_droid_trajectories(pattern_or_path)
        else:
            # Local pattern scanning
            return glob.glob(pattern_or_path)
            
    def _find_local_trajectories(self, directory: str) -> List[str]:
        """Find DROID trajectories in a local directory."""
        trajectories = []
        for root, dirs, files in os.walk(directory):
            if "trajectory.h5" in files and "recordings" in dirs:
                trajectories.append(root)
        return trajectories
        
    def _create_initial_dataset(self) -> Dataset:
        """Create initial Ray dataset with trajectory metadata."""
        trajectory_items = []
        
        for i, traj_path in enumerate(self.trajectory_paths):
            # Extract trajectory metadata from path
            traj_name = traj_path.rstrip("/").split("/")[-1]
            is_gcs = traj_path.startswith("gs://")
            
            # Infer success/failure from path
            success_label = None
            if "success" in traj_path.lower():
                success_label = True
            elif "failure" in traj_path.lower():
                success_label = False
                
            item = {
                "trajectory_id": i,
                "trajectory_name": traj_name,
                "trajectory_path": traj_path,
                "is_gcs": is_gcs,
                "success_label": success_label,
                "local_path": None,  # Will be populated after download
                "__metadata_only__": True  # Indicates this is metadata only
            }
            trajectory_items.append(item)
            
        return rd.from_items(trajectory_items)
    
    def _download_trajectory(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Download a single DROID trajectory if needed."""
        traj_path = item["trajectory_path"]
        
        if not item["is_gcs"]:
            # Already local
            item["local_path"] = traj_path
            item["__metadata_only__"] = False
            return item
            
        # Download from GCS if needed
        if traj_path not in self._downloaded_paths:
            from examples.droid_h5.droid_hdf5_pipeline import download_droid_trajectory
            
            success, local_path, error_msg, traj_name = ray.get(
                download_droid_trajectory.remote(
                    traj_path, 
                    self.local_dir, 
                    tempfile.mkdtemp()
                )
            )
            
            if success:
                self._downloaded_paths[traj_path] = local_path
            else:
                logger.error(f"Failed to download {traj_path}: {error_msg}")
                item["error"] = error_msg
                return item
                
        item["local_path"] = self._downloaded_paths[traj_path]
        item["__metadata_only__"] = False
        return item
        
    def _load_trajectory_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Load full trajectory data using DROID backend."""
        if item.get("__metadata_only__", True):
            # Download first if needed
            item = self._download_trajectory(item)
            
        if "error" in item or not item.get("local_path"):
            return item
            
        try:
            # Load trajectory using simple loader
            local_path = item["local_path"]
            trajectory_data = load_droid_trajectory_simple(local_path)
            
            # Merge trajectory data with metadata
            result = {**item}
            result.update(trajectory_data)
            result["__trajectory_loaded__"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading trajectory {item['trajectory_name']}: {e}")
            item["error"] = str(e)
            return item
    
    def _ensure_downloaded(self):
        """Ensure all trajectories are downloaded (if GCS)."""
        if self._is_downloaded:
            return
            
        # Download all GCS trajectories in parallel
        gcs_items = [item for item in self.ray_dataset.take_all() if item.get("is_gcs", False)]
        
        if gcs_items:
            logger.info(f"Downloading {len(gcs_items)} DROID trajectories...")
            self.ray_dataset = self.ray_dataset.map(
                self._download_trajectory,
                num_cpus=self.config.download_workers
            )
            
        self._is_downloaded = True
        
    def load_trajectories(self):
        """Load trajectory data for all trajectories."""
        self._ensure_downloaded()
        
        # Create new dataset with loaded trajectory data
        loaded_dataset = DroidDataset.__new__(DroidDataset)
        loaded_dataset.config = self.config
        loaded_dataset.trajectory_paths = self.trajectory_paths
        loaded_dataset.local_dir = self.local_dir
        loaded_dataset._downloaded_paths = self._downloaded_paths
        loaded_dataset._is_downloaded = True
        
        # Load trajectory data
        loaded_dataset.ray_dataset = self.ray_dataset.map(
            self._load_trajectory_data,
            num_cpus=self.config.num_parallel_reads
        )
        
        return loaded_dataset
        
    def filter(self, fn):
        """Filter trajectories with lazy loading."""
        # Create filtered dataset
        filtered_dataset = DroidDataset.__new__(DroidDataset)
        filtered_dataset.config = self.config
        filtered_dataset.trajectory_paths = self.trajectory_paths
        filtered_dataset.local_dir = self.local_dir
        filtered_dataset._downloaded_paths = self._downloaded_paths
        filtered_dataset._is_downloaded = self._is_downloaded
        
        # Apply filter with automatic data loading
        def load_and_filter(item):
            # Load trajectory data if needed for filtering
            if item.get("__metadata_only__", True):
                loaded_item = self._load_trajectory_data(item)
            else:
                loaded_item = item
                
            # Apply filter function
            if "error" in loaded_item:
                return {"__keep__": False, **loaded_item}
            
            try:
                keep = fn(loaded_item)
                return {"__keep__": bool(keep), **loaded_item}
            except Exception as e:
                logger.warning(f"Filter function failed for {loaded_item.get('trajectory_name', 'unknown')}: {e}")
                return {"__keep__": False, **loaded_item}
                
        # Apply combined load-and-filter operation
        temp_dataset = self.ray_dataset.map(
            load_and_filter,
            num_cpus=self.config.num_parallel_reads
        )
        
        # Filter based on __keep__ flag and remove it
        filtered_dataset.ray_dataset = temp_dataset.filter(
            lambda item: item.get("__keep__", False)
        ).map(
            lambda item: {k: v for k, v in item.items() if k != "__keep__"}
        )
        
        return filtered_dataset
        
    def map(self, fn, **kwargs):
        """Map function over trajectories with lazy loading."""
        mapped_dataset = DroidDataset.__new__(DroidDataset)
        mapped_dataset.config = self.config
        mapped_dataset.trajectory_paths = self.trajectory_paths
        mapped_dataset.local_dir = self.local_dir
        mapped_dataset._downloaded_paths = self._downloaded_paths
        mapped_dataset._is_downloaded = self._is_downloaded
        
        def load_and_map(item):
            # Load trajectory data if needed
            if item.get("__metadata_only__", True):
                loaded_item = self._load_trajectory_data(item)
            else:
                loaded_item = item
                
            if "error" in loaded_item:
                return loaded_item
                
            try:
                return fn(loaded_item)
            except Exception as e:
                logger.warning(f"Map function failed for {loaded_item.get('trajectory_name', 'unknown')}: {e}")
                loaded_item["error"] = str(e)
                return loaded_item
        
        # Use provided kwargs or defaults
        if 'num_cpus' not in kwargs:
            kwargs['num_cpus'] = self.config.num_parallel_reads
        
        mapped_dataset.ray_dataset = self.ray_dataset.map(load_and_map, **kwargs)
        return mapped_dataset
        
    # Ray Dataset compatibility methods
    def get_ray_dataset(self) -> Dataset:
        """Get the underlying Ray dataset."""
        return self.ray_dataset
        
    def count(self) -> int:
        """Count trajectories in dataset."""
        return self.ray_dataset.count()
        
    def take(self, num_items: int) -> List[Dict[str, Any]]:
        """Take specified number of items."""
        return self.ray_dataset.take(num_items)
        
    def take_all(self) -> List[Dict[str, Any]]:
        """Take all items."""
        return self.ray_dataset.take_all()
        
    def schema(self):
        """Get dataset schema."""
        return self.ray_dataset.schema()
        
    def iter_batches(self, batch_size: int = 1):
        """Iterate over batches."""
        return self.ray_dataset.iter_batches(batch_size=batch_size)
        
    def iter_rows(self):
        """Iterate over rows."""
        return self.ray_dataset.iter_rows()
        
    def materialize(self):
        """Materialize the dataset."""
        return self.ray_dataset.materialize()
    
    def __len__(self) -> int:
        return self.count()
        
    def __repr__(self) -> str:
        return f"DroidDataset(trajectories={len(self.trajectory_paths)}, downloaded={self._is_downloaded})"


def load_droid_dataset(
    trajectory_paths: Union[str, List[str]],
    local_dir: Optional[str] = None,
    auto_download: bool = True,
    **kwargs
) -> DroidDataset:
    """
    Load a DROID dataset from trajectory paths.
    
    Args:
        trajectory_paths: GCS paths, local paths, or patterns for DROID trajectories
        local_dir: Local directory for downloads
        auto_download: Whether to auto-download GCS trajectories
        **kwargs: Additional configuration options
        
    Returns:
        DroidDataset instance
        
    Example:
        >>> # Load from GCS pattern
        >>> dataset = load_droid_dataset("gs://gresearch/robotics/droid_raw/1.0.1/RAIL/success/*")
        >>> 
        >>> # Load specific trajectories  
        >>> paths = ["gs://path/to/traj1", "gs://path/to/traj2"]
        >>> dataset = load_droid_dataset(paths)
        >>>
        >>> # Use with Agent
        >>> from robodm.agent import Agent
        >>> agent = Agent(dataset)
        >>> filtered = agent.filter("trajectories that are successful")
    """
    config = DroidDatasetConfig(auto_download=auto_download, **kwargs)
    return DroidDataset(trajectory_paths, local_dir, config)