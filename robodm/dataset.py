import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Text

import numpy as np

try:
    import ray
    import ray.data as rd

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import robodm
from robodm.metadata.metadata_manager import MetadataManager
from robodm.utils.flatten import data_to_tf_schema

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for VLADataset."""

    batch_size: int = 1
    shuffle: bool = False
    num_parallel_reads: int = 128
    ray_init_kwargs: Optional[Dict] = None
    use_metadata: bool = True
    auto_build_metadata: bool = True


class VLADataset:
    """
    Ray Dataset-based VLA dataset with integrated metadata management.
    
    This dataset integrates:
    1. Ray Dataset for parallel data loading and processing
    2. MetadataManager for efficient metadata handling
    3. Automatic data management and optimization
    """

    def __init__(
        self,
        path: Text,
        return_type: str = "numpy",
        config: Optional[DatasetConfig] = None,
        **kwargs,
    ):
        """
        Initialize VLA dataset.

        Args:
            path: Path to VLA files (can be glob pattern, directory, or single file)
            return_type: Return type ("numpy", "tensor", "container")
            config: Dataset configuration
            **kwargs: Additional arguments
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for VLADataset. Install with: pip install 'ray[data]'"
            )

        self.path = path
        self.return_type = return_type
        self.config = config or DatasetConfig()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**(self.config.ray_init_kwargs or {}))

        # Get file paths and create Ray dataset
        self.file_paths = self._get_files(path)
        self.ray_dataset = self._create_dataset()

        # Initialize metadata manager
        self.metadata_manager = self._create_metadata_manager()

        # Cache for schema and stats
        self._schema = None
        self._stats: Optional[Dict[str, Any]] = None
        
        # Track dataset state - starts with just file paths
        self._is_loaded = False
        self._has_file_paths = True
        
        logger.info(f"Initialized VLADataset with {len(self.file_paths)} files")

    def _get_files(self, path: str) -> List[str]:
        """Get list of VLA files based on path."""
        files = []

        if "*" in path:
            files = glob.glob(path)
        elif os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.vla"))
        else:
            files = [path]

        return files

    def _create_dataset(self) -> rd.Dataset:
        """Create Ray dataset from file paths."""
        # Create dataset from file paths
        dataset = rd.from_items(self.file_paths)

        # Apply shuffling if requested
        if self.config.shuffle:
            dataset = dataset.random_shuffle()

        return dataset

    def _load_trajectory(self, item) -> Dict[str, Any]:
        """Load a complete trajectory from file."""
        # Handle both string paths and dict items from Ray dataset
        if isinstance(item, dict):
            file_path = item.get("item", item)
        else:
            file_path = item

        try:
            traj = robodm.Trajectory(file_path)
            data = traj.load(return_type=self.return_type)
            # Add file path metadata for tracking
            data["__file_path__"] = str(file_path)
            
            return data
        except Exception as e:
            logger.error(f"Error loading trajectory {file_path}: {e}")
            return {"__file_path__": str(file_path)}

    def _create_metadata_manager(self) -> Optional[MetadataManager]:
        """Create and initialize metadata manager."""
        if not self.config.use_metadata:
            return None
            
        # Create metadata manager that works with ray dataset
        manager = MetadataManager.from_ray_dataset(
            self.ray_dataset,
            auto_build=self.config.auto_build_metadata
        )
        
        return manager

    def get_ray_dataset(self) -> rd.Dataset:
        """Get the underlying Ray dataset.
        
        Note: If dataset is not loaded, this returns a dataset of file paths.
        Consider using filter() or map() methods which handle loading automatically.
        """
        if not self._is_loaded and self._has_file_paths:
            logger.warning("Accessing Ray dataset with file paths only. "
                         "Consider using VLADataset methods for automatic loading.")
        return self.ray_dataset

    def iter_batches(self, batch_size: Optional[int] = None):
        """Iterate over batches of data."""
        batch_size = batch_size or self.config.batch_size
        return self.ray_dataset.iter_batches(batch_size=batch_size)

    def iter_rows(self):
        """Iterate over individual rows of data."""
        return self.ray_dataset.iter_rows()

    def take(self, num_items: int) -> List[Dict[str, Any]]:
        """Take a specific number of items."""
        return list(self.ray_dataset.take(num_items))

    def sample(self,
               num_samples: int,
               replace: bool = False) -> List[Dict[str, Any]]:
        """Sample from the dataset."""
        total_count = self.count()
        if total_count == 0:
            return []

        if not replace:
            shuffled_dataset = self.ray_dataset.random_shuffle()
            return list(shuffled_dataset.take(min(num_samples, total_count)))
        else:
            import warnings
            warnings.warn(
                "Sampling with replacement may not return exact count due to Ray API limitations"
            )
            fraction = min(1.0, num_samples / total_count)
            sampled = self.ray_dataset.random_sample(fraction)
            return list(sampled.take(num_samples))

    def count(self) -> int:
        """Count the number of items in the dataset."""
        return self.ray_dataset.count()

    def schema(self):
        """Get the schema of the dataset."""
        if self._schema is None:
            self._schema = self.ray_dataset.schema()
        return self._schema

    def split(self, *fractions: float, shuffle: bool = True):
        """Split the dataset into multiple datasets."""
        # Validate fractions sum to <= 1.0
        if sum(fractions) > 1.0:
            raise ValueError(
                f"Sum of fractions {sum(fractions)} must be <= 1.0")

        # Ray Dataset.split() doesn't support shuffle parameter
        dataset_to_split = self.ray_dataset.random_shuffle() if shuffle else self.ray_dataset

        if len(fractions) == 1:
            ray_datasets = dataset_to_split.train_test_split(test_size=fractions[0], shuffle=False)
        elif len(fractions) == 2 and abs(sum(fractions) - 1.0) < 1e-10:
            ray_datasets = dataset_to_split.train_test_split(test_size=fractions[1], shuffle=False)
        else:
            fractions_list = list(fractions)
            total = sum(fractions_list)

            if abs(total - 1.0) < 1e-10:
                fractions_list[-1] -= 1e-6
                splits = dataset_to_split.split_proportionately(fractions_list)
                ray_datasets = splits[:-1]
            else:
                ray_datasets = dataset_to_split.split_proportionately(fractions_list)

        # Create new VLADataset instances for each split
        split_datasets = []
        for ray_ds in ray_datasets:
            split_dataset = VLADataset.__new__(VLADataset)
            split_dataset.path = self.path
            split_dataset.return_type = self.return_type
            split_dataset.config = self.config
            split_dataset.file_paths = self.file_paths
            split_dataset.ray_dataset = ray_ds
            split_dataset.metadata_manager = self.metadata_manager
            split_dataset._schema = self._schema
            split_dataset._stats = None
            split_dataset._is_loaded = self._is_loaded
            split_dataset._has_file_paths = self._has_file_paths
            split_datasets.append(split_dataset)

        return split_datasets

    def _ensure_loaded(self):
        """Ensure trajectories are loaded, applying lazy loading if needed."""
        if not self._is_loaded and self._has_file_paths:
            # Apply lazy loading transformation
            self.ray_dataset = self.ray_dataset.map(
                self._load_trajectory,
                num_cpus=self.config.num_parallel_reads,
                concurrency=self.config.num_parallel_reads,
            )
            self._is_loaded = True
            logger.info("Applied lazy trajectory loading transformation")
    
    def filter(self, fn):
        """Filter the dataset with automatic lazy loading."""
        filtered_dataset = VLADataset.__new__(VLADataset)
        filtered_dataset.path = self.path
        filtered_dataset.return_type = self.return_type
        filtered_dataset.config = self.config
        filtered_dataset.file_paths = self.file_paths
        
        # Handle lazy loading - don't load if not needed
        if not self._is_loaded and self._has_file_paths:
            # Create a combined load-and-filter operation for efficiency
            def load_and_filter(item):
                trajectory = self._load_trajectory(item)
                # Add filter result as a field in the trajectory
                keep = fn(trajectory)
                trajectory['__filter_result__'] = keep
                return trajectory
            
            # Apply combined operation
            temp_dataset = self.ray_dataset.map(
                load_and_filter,
                num_cpus=self.config.num_parallel_reads,
                concurrency=self.config.num_parallel_reads,
            )
            
            # Filter based on the result and remove the temporary field
            filtered_dataset.ray_dataset = temp_dataset.filter(
                lambda item: item['__filter_result__']
            ).map(lambda item: {k: v for k, v in item.items() if k != '__filter_result__'})
            
            filtered_dataset._is_loaded = True
        else:
            # Already loaded, just filter normally
            filtered_dataset.ray_dataset = self.ray_dataset.filter(fn)
            filtered_dataset._is_loaded = self._is_loaded
        
        filtered_dataset._has_file_paths = self._has_file_paths
        filtered_dataset.metadata_manager = self.metadata_manager
        filtered_dataset._schema = self._schema
        filtered_dataset._stats = None
        return filtered_dataset

    def map(self, fn, **kwargs):
        """Map a function over the dataset with automatic lazy loading."""
        mapped_dataset = VLADataset.__new__(VLADataset)
        mapped_dataset.path = self.path
        mapped_dataset.return_type = self.return_type
        mapped_dataset.config = self.config
        mapped_dataset.file_paths = self.file_paths
        
        # Handle lazy loading
        if not self._is_loaded and self._has_file_paths:
            # Combine load and map operations
            def load_and_map(item):
                trajectory = self._load_trajectory(item)
                return fn(trajectory)
            
            # Use provided kwargs or default to config settings
            if 'num_cpus' not in kwargs:
                kwargs['num_cpus'] = self.config.num_parallel_reads
            if 'concurrency' not in kwargs:
                kwargs['concurrency'] = self.config.num_parallel_reads
                
            mapped_dataset.ray_dataset = self.ray_dataset.map(load_and_map, **kwargs)
            mapped_dataset._is_loaded = True
        else:
            # Already loaded, just map normally
            mapped_dataset.ray_dataset = self.ray_dataset.map(fn, **kwargs)
            mapped_dataset._is_loaded = self._is_loaded
            
        mapped_dataset._has_file_paths = self._has_file_paths
        mapped_dataset.metadata_manager = self.metadata_manager
        mapped_dataset._schema = None  # Schema might change after mapping
        mapped_dataset._stats = None
        return mapped_dataset
    
    def load_trajectories(self):
        """Load trajectory data from file paths using map function."""
        if self._is_loaded:
            logger.info("Dataset already loaded, returning self")
            return self
            
        loaded_dataset = VLADataset.__new__(VLADataset)
        loaded_dataset.path = self.path
        loaded_dataset.return_type = self.return_type
        loaded_dataset.config = self.config
        loaded_dataset.file_paths = self.file_paths
        
        # Apply loading transformation
        loaded_dataset.ray_dataset = self.ray_dataset.map(
            self._load_trajectory,
            num_cpus=self.config.num_parallel_reads,
            concurrency=self.config.num_parallel_reads,
        )
        
        # Update state
        loaded_dataset._is_loaded = True
        loaded_dataset._has_file_paths = self._has_file_paths
        loaded_dataset.metadata_manager = self.metadata_manager
        loaded_dataset._schema = None
        loaded_dataset._stats = None
        
        return loaded_dataset
    
    def _select_frame(self, item, frame_type: str = "last") -> Dict[str, Any]:
        """Select a specific frame from trajectory data at query time."""
        # Handle both string paths and loaded trajectory data
        if isinstance(item, str) or (isinstance(item, dict) and "__file_path__" not in item):
            # Load trajectory if not already loaded
            trajectory_data = self._load_trajectory(item)
        else:
            trajectory_data = item
            
        # Find camera/image keys
        camera_keys = [k for k in trajectory_data.keys() if "observation/images/" in k or "image" in k.lower()]
        
        result = {}
        
        # Copy non-trajectory data (metadata, etc.) and preserve trajectory metadata
        for key, value in trajectory_data.items():
            if key.startswith("__") or key not in camera_keys:
                result[key] = value
        
        # Preserve additional trajectory metadata
        if "__file_path__" in trajectory_data:
            result["__file_path__"] = trajectory_data["__file_path__"]
            result["__frame_type__"] = frame_type
        
        # Select frames based on frame_type
        for camera_key in camera_keys:
            frames = trajectory_data.get(camera_key, [])
            if len(frames) == 0:
                result[camera_key] = None
                continue
                
            if frame_type == "first":
                result[camera_key] = frames[0]
            elif frame_type == "middle":
                result[camera_key] = frames[len(frames) // 2]
            elif frame_type == "last":
                result[camera_key] = frames[-1]
            else:
                # Return all frames by default
                result[camera_key] = frames
                
        return result
    
    def select_frames(self, frame_type: str = "last"):
        """Create a dataset with selected frames at query time."""
        return self.map(lambda item: self._select_frame(item, frame_type))

    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the dataset."""
        shuffled_dataset = VLADataset.__new__(VLADataset)
        shuffled_dataset.path = self.path
        shuffled_dataset.return_type = self.return_type
        shuffled_dataset.config = self.config
        shuffled_dataset.file_paths = self.file_paths
        shuffled_dataset.ray_dataset = self.ray_dataset.random_shuffle(seed=seed)
        shuffled_dataset.metadata_manager = self.metadata_manager
        shuffled_dataset._schema = self._schema
        shuffled_dataset._stats = None
        shuffled_dataset._is_loaded = self._is_loaded
        shuffled_dataset._has_file_paths = self._has_file_paths
        return shuffled_dataset

    def materialize(self):
        """Materialize the dataset in memory."""
        return self.ray_dataset.materialize()

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self._stats is None:
            sample = self.peek()
            if sample:
                self._stats = {
                    "return_type": self.return_type,
                    "total_items": self.count(),
                    "sample_keys": (list(sample.keys()) if isinstance(sample, dict) else []),
                }

                # Add trajectory length info from first data key (excluding metadata)
                data_keys = [k for k in sample.keys() if not k.startswith("__")]
                if data_keys and sample:
                    first_key = data_keys[0]
                    if hasattr(sample[first_key], "__len__"):
                        self._stats["trajectory_length"] = len(sample[first_key])
            else:
                self._stats = {"total_items": 0}

        return self._stats

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the first item without consuming it."""
        try:
            return self.ray_dataset.take(1)[0]
        except:
            return None

    def get_tf_schema(self):
        """Get TensorFlow schema for the dataset."""
        sample = self.peek()
        if sample:
            # Filter out metadata keys
            data_sample = {k: v for k, v in sample.items() if not k.startswith("__")}
            return data_to_tf_schema(data_sample)
        return None

    def __iter__(self):
        """Iterate over the dataset."""
        return self.iter_rows()

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return self.count()

    def __getitem__(self, index):
        """Not supported for Ray datasets - use take() or sample() instead."""
        raise NotImplementedError(
            "Random access not supported for Ray datasets. "
            "Use take(), sample(), or iterate over the dataset instead.")


# Utility functions for common dataset operations
def load_dataset(
    path: Text,
    return_type: str = "numpy",
    batch_size: int = 1,
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    **kwargs,
) -> VLADataset:
    """Load a VLA dataset from path."""
    config = DatasetConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads
    )
    return VLADataset(
        path=path,
        return_type=return_type,
        config=config,
        **kwargs
    )


def split_dataset(
    dataset: VLADataset,
    train_fraction: float = 0.8,
    val_fraction: float = 0.2,
    shuffle: bool = False,
) -> tuple[VLADataset, VLADataset]:
    """Split a dataset into train and validation sets."""
    if abs(train_fraction + val_fraction - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction must equal 1.0")

    splits = dataset.split(train_fraction, val_fraction, shuffle=shuffle)
    return splits[0], splits[1]