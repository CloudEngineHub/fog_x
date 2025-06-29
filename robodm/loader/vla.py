import glob
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

import numpy as np

try:
    import ray
    import ray.data as rd

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import robodm
from robodm.loader.base import BaseLoader
from robodm.metadata_manager import MetadataManager, TrajectoryMetadata
from robodm.metadata_utils import build_dataset_metadata

logger = logging.getLogger(__name__)


class LoadingMode(Enum):
    """Loading mode for the VLA loader."""

    TRAJECTORY = "trajectory"  # Load entire trajectories
    SLICE = "slice"  # Load random slices from trajectories


@dataclass
class SliceConfig:
    """Configuration for slice loading mode."""

    slice_length: int = 100  # Number of timesteps per slice
    min_slice_length: Optional[int] = (
        None  # Minimum slice length (defaults to slice_length)
    )
    stride: int = 1  # Stride between consecutive timesteps in slice
    random_start: bool = True  # Whether to randomly sample start position
    overlap_ratio: float = 0.0  # Overlap ratio between consecutive slices (0.0-1.0)


class RayVLALoader(BaseLoader):
    """
    Ray Dataset-based VLA loader supporting both trajectory and slice loading modes.

    This loader uses Ray Dataset for parallel data loading, automatic shuffling,
    and efficient data splitting.
    """

    def __init__(
        self,
        path: Text,
        mode: LoadingMode = LoadingMode.TRAJECTORY,
        batch_size: int = 1,
        return_type: str = "numpy",
        shuffle: bool = False,
        num_parallel_reads: int = 4,
        slice_config: Optional[SliceConfig] = None,
        ray_init_kwargs: Optional[Dict] = None,
        use_metadata: bool = True,
        auto_build_metadata: bool = True,
    ):
        """
        Initialize the Ray VLA loader.

        Args:
            path: Path to VLA files (can be glob pattern, directory, or single file)
            mode: Loading mode (TRAJECTORY or SLICE)
            batch_size: Batch size for data loading
            return_type: Return type ("numpy", "tensor", "container")
            shuffle: Whether to shuffle the data
            num_parallel_reads: Number of parallel read operations
            slice_config: Configuration for slice mode (required if mode=SLICE)
            ray_init_kwargs: Additional kwargs for Ray initialization
            use_metadata: Whether to use parquet metadata files for efficient loading
            auto_build_metadata: Whether to automatically build metadata if missing
        """
        super().__init__(path)

        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for RayVLALoader. Install with: pip install 'ray[data]'"
            )

        self.mode = mode
        self.batch_size = batch_size
        self.return_type = return_type
        self.shuffle = shuffle
        self.num_parallel_reads = num_parallel_reads
        self.slice_config = slice_config or SliceConfig()
        self.use_metadata = use_metadata
        self.auto_build_metadata = auto_build_metadata

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**(ray_init_kwargs or {}))

        # Validate slice config for slice mode
        if mode == LoadingMode.SLICE and slice_config is None:
            self.slice_config = SliceConfig()

        # Initialize metadata manager if using metadata
        self.metadata_manager: Optional[MetadataManager] = None
        self.metadata_cache: Dict[str, Any] = {}
        if self.use_metadata:
            self._initialize_metadata()

        # Get file paths and create Ray dataset
        self.file_paths = self._get_files(path)
        self.dataset = self._create_dataset()

        logger.info(
            f"Initialized RayVLALoader with {len(self.file_paths)} files in {mode.value} mode"
        )

    def _initialize_metadata(self):
        """Initialize metadata manager and build metadata if needed."""
        # Determine the dataset directory
        path_obj = Path(self.path)
        if path_obj.is_dir():
            dataset_dir = path_obj
        elif "*" in self.path:
            # For glob patterns, use the parent directory
            dataset_dir = Path(self.path).parent
        else:
            # For single file, use its parent directory
            dataset_dir = path_obj.parent

        self.metadata_manager = MetadataManager(dataset_dir)

        # Check if metadata exists
        if not self.metadata_manager.exists():
            if self.auto_build_metadata:
                logger.info(f"Building metadata for dataset at {dataset_dir}")
                build_dataset_metadata(str(dataset_dir))
            else:
                logger.warning(
                    "Metadata file not found and auto_build_metadata is False")
                self.use_metadata = False
                return

        # Load metadata into cache
        try:
            all_metadata = self.metadata_manager.get_all_metadata()
            self.metadata_cache = {
                meta.file_path: meta
                for meta in all_metadata
            }
            logger.info(
                f"Loaded metadata for {len(self.metadata_cache)} trajectories")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.use_metadata = False

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
        """Create Ray dataset based on loading mode."""
        # Create initial dataset from file paths
        dataset = rd.from_items(self.file_paths)

        if self.mode == LoadingMode.TRAJECTORY:
            # For trajectory mode, each item is a complete trajectory
            dataset = dataset.map(
                self._load_trajectory,
                num_cpus=self.num_parallel_reads,
                concurrency=self.num_parallel_reads,
            )
        elif self.mode == LoadingMode.SLICE:
            # For slice mode, expand each trajectory into multiple slices
            dataset = dataset.flat_map(
                self._extract_slices,
                num_cpus=self.num_parallel_reads,
                concurrency=self.num_parallel_reads,
            )

        # Apply shuffling if requested
        if self.shuffle:
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

            return data

        except Exception as e:
            logger.error(f"Error loading trajectory {file_path}: {e}")
            return {}

    def _extract_slices(self, item) -> List[Dict[str, Any]]:
        """Extract slices from a trajectory file."""
        # Handle both string paths and dict items from Ray dataset
        if isinstance(item, dict):
            file_path = item.get("item", item)
        else:
            file_path = item

        try:
            # Try to get trajectory length from metadata first
            file_path_str = str(Path(file_path).resolve())
            traj_length = None

            if self.use_metadata and file_path_str in self.metadata_cache:
                metadata = self.metadata_cache[file_path_str]
                traj_length = metadata.trajectory_length
                logger.debug(
                    f"Using cached metadata for {file_path}: length={traj_length}"
                )

            # If we have metadata and know the trajectory is too short, skip loading
            min_length = (self.slice_config.min_slice_length
                          or self.slice_config.slice_length)

            if traj_length is not None and traj_length < min_length:
                logger.warning(
                    f"Trajectory {file_path} too short ({traj_length} < {min_length})"
                )
                return []

            # Load trajectory data
            traj = robodm.Trajectory(file_path)
            full_data = traj.load(return_type=self.return_type)

            if not full_data:
                return []

            # Get trajectory length if we didn't have it from metadata
            if traj_length is None:
                traj_length = len(next(iter(full_data.values())))

            slices = []
            slice_step = max(
                1,
                int(self.slice_config.slice_length *
                    (1 - self.slice_config.overlap_ratio)),
            )

            # Generate slice positions
            max_start = traj_length - self.slice_config.slice_length

            if self.slice_config.random_start:
                # Random sampling of slice positions
                num_slices = max(1, max_start // slice_step)
                start_positions = [
                    random.randint(0, max_start) for _ in range(num_slices)
                ]
            else:
                # Sequential slicing
                start_positions = list(range(0, max_start + 1, slice_step))

            # Extract slices
            for start_idx in start_positions:
                end_idx = min(start_idx + self.slice_config.slice_length,
                              traj_length)
                actual_length = end_idx - start_idx

                if actual_length < min_length:
                    continue

                # Extract slice data
                slice_data = {}
                for key, values in full_data.items():
                    if isinstance(values, np.ndarray):
                        slice_data[key] = values[start_idx:end_idx:self.
                                                 slice_config.stride]
                    elif isinstance(values, list):
                        slice_data[key] = values[start_idx:end_idx:self.
                                                 slice_config.stride]
                    else:
                        slice_data[key] = values

                slices.append(slice_data)

            return slices

        except Exception as e:
            logger.error(f"Error extracting slices from {file_path}: {e}")
            return []

    def get_batch(self) -> List[Dict[str, Any]]:
        """Get a batch of data."""
        try:
            batch = self.dataset.take(self.batch_size)
            return list(batch)
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            return []

    def iter_batches(self, batch_size: Optional[int] = None):
        """Iterate over batches of data."""
        batch_size = batch_size or self.batch_size
        return self.dataset.iter_batches(batch_size=batch_size)

    def iter_rows(self):
        """Iterate over individual rows of data."""
        return self.dataset.iter_rows()

    def take(self, num_items: int) -> List[Dict[str, Any]]:
        """Take a specific number of items."""
        return list(self.dataset.take(num_items))

    def count(self) -> int:
        """Count the number of items in the dataset."""
        return self.dataset.count()

    def schema(self):
        """Get the schema of the dataset."""
        return self.dataset.schema()

    def split(self, *fractions: float, shuffle: bool = True):
        """Split the dataset into multiple datasets."""
        # Validate fractions sum to <= 1.0
        if sum(fractions) > 1.0:
            raise ValueError(
                f"Sum of fractions {sum(fractions)} must be <= 1.0")

        # Ray Dataset.split() doesn't support shuffle parameter
        # If shuffle is requested, shuffle the dataset first
        dataset_to_split = self.dataset.random_shuffle(
        ) if shuffle else self.dataset

        if len(fractions) == 1:
            # For single fraction, convert to train/test split
            return dataset_to_split.train_test_split(test_size=fractions[0],
                                                     shuffle=False)
        elif len(fractions) == 2 and abs(sum(fractions) - 1.0) < 1e-10:
            # Special case: exactly two fractions that sum to 1.0
            # Use train_test_split which handles this case
            return dataset_to_split.train_test_split(test_size=fractions[1],
                                                     shuffle=False)
        else:
            # For multiple fractions, use split_proportionately
            # Ray requires the sum to be < 1.0, so if it equals 1.0, we need to adjust
            fractions_list = list(fractions)
            total = sum(fractions_list)

            if abs(total - 1.0) < 1e-10:
                # If fractions sum to 1.0, subtract a tiny amount from the last fraction
                # so Ray doesn't complain, then drop the extra split
                fractions_list[-1] -= 1e-6
                splits = dataset_to_split.split_proportionately(fractions_list)
                # Drop the last split (which will be nearly empty)
                return splits[:-1]
            else:
                return dataset_to_split.split_proportionately(fractions_list)

    def filter(self, fn):
        """Filter the dataset."""
        return self.dataset.filter(fn)

    def map(self, fn, **kwargs):
        """Map a function over the dataset."""
        return self.dataset.map(fn, **kwargs)

    def sample(self, num_samples: int, replace: bool = False):
        """Sample from the dataset."""
        # Ray's random_sample expects a fraction, not absolute count
        total_count = self.count()
        if total_count == 0:
            return []

        # For exact count without replacement, use take with random shuffle
        if not replace:
            shuffled_dataset = self.dataset.random_shuffle()
            return list(shuffled_dataset.take(min(num_samples, total_count)))
        else:
            # For replacement sampling, use multiple passes if needed
            # This is a limitation of Ray's API
            import warnings

            warnings.warn(
                "Sampling with replacement may not return exact count due to Ray API limitations"
            )

            fraction = min(1.0, num_samples / total_count)
            # Sample and take up to the requested amount
            sampled = self.dataset.random_sample(fraction)
            return list(sampled.take(num_samples))

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the first item without consuming it."""
        try:
            return self.dataset.take(1)[0]
        except:
            return None

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return self.count()

    def __iter__(self):
        """Iterate over the dataset."""
        return self.iter_rows()

    def materialize(self):
        """Materialize the dataset in memory."""
        return self.dataset.materialize()


# Legacy compatibility loaders (deprecated)
class VLALoader(RayVLALoader):
    """Legacy VLA loader - deprecated, use RayVLALoader instead."""

    def __init__(self, path: Text, batch_size=1, return_type="numpy"):
        logger.warning("VLALoader is deprecated. Use RayVLALoader instead.")
        super().__init__(
            path=path,
            mode=LoadingMode.TRAJECTORY,
            batch_size=batch_size,
            return_type=return_type,
            shuffle=True,
        )


class NonShuffleVLALoader(RayVLALoader):
    """Legacy non-shuffle VLA loader - deprecated, use RayVLALoader instead."""

    def __init__(self,
                 path: Text,
                 batch_size=1,
                 num_workers=1,
                 return_type="numpy"):
        logger.warning(
            "NonShuffleVLALoader is deprecated. Use RayVLALoader instead.")
        super().__init__(
            path=path,
            mode=LoadingMode.TRAJECTORY,
            batch_size=batch_size,
            return_type=return_type,
            shuffle=False,
        )


def get_vla_dataloader(path: Text,
                       batch_size: int = 1,
                       num_workers: int = 1,
                       **kwargs):
    """Legacy function to get VLA dataloader - deprecated, use create_trajectory_loader instead."""
    logger.warning(
        "get_vla_dataloader is deprecated. Use create_trajectory_loader instead."
    )
    loader = RayVLALoader(
        path=path,
        mode=LoadingMode.TRAJECTORY,
        batch_size=batch_size,
        return_type="numpy",
        shuffle=True,
        num_parallel_reads=max(1, num_workers),
        **kwargs,
    )
    return loader


# Factory functions for common use cases
def create_trajectory_loader(
    path: Text,
    batch_size: int = 1,
    return_type: str = "numpy",
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    **kwargs,
) -> RayVLALoader:
    """Create a loader for complete trajectories."""
    return RayVLALoader(
        path=path,
        mode=LoadingMode.TRAJECTORY,
        batch_size=batch_size,
        return_type=return_type,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads,
        **kwargs,
    )


def create_slice_loader(
    path: Text,
    slice_length: int = 100,
    batch_size: int = 1,
    return_type: str = "numpy",
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    min_slice_length: Optional[int] = None,
    stride: int = 1,
    random_start: bool = True,
    overlap_ratio: float = 0.0,
    **kwargs,
) -> RayVLALoader:
    """Create a loader for trajectory slices."""
    slice_config = SliceConfig(
        slice_length=slice_length,
        min_slice_length=min_slice_length,
        stride=stride,
        random_start=random_start,
        overlap_ratio=overlap_ratio,
    )

    return RayVLALoader(
        path=path,
        mode=LoadingMode.SLICE,
        batch_size=batch_size,
        return_type=return_type,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads,
        slice_config=slice_config,
        **kwargs,
    )
