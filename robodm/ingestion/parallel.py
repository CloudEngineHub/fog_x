"""
Ray-based parallel processing for data ingestion.

This module provides the core parallel processing capabilities using Ray
to efficiently transform data sources into robodm trajectories.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .base import BatchProcessor, DataIngestionInterface, IngestionConfig

logger = logging.getLogger(__name__)


@ray.remote
class TrajectoryWorker:
    """Ray actor for processing trajectory groups in parallel."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize worker with configuration."""
        # Reconstruct config from dict
        self.config = IngestionConfig(**config_dict)
        self.processor = None

    def initialize_processor(self, ingester_class: type,
                             ingester_kwargs: Dict[str, Any]):
        """Initialize the batch processor with the ingester."""
        ingester = ingester_class(**ingester_kwargs)
        self.processor = BatchProcessor(ingester, self.config)

    def process_batch(self, trajectory_groups: List[List[Any]]) -> List[str]:
        """Process a batch of trajectory groups."""
        if self.processor is None:
            raise RuntimeError("Worker not initialized")

        return self.processor.process_trajectory_groups(trajectory_groups)


class ParallelDataIngester:
    """
    Ray-based parallel data ingestion engine.

    This class coordinates the parallel transformation of data sources
    into robodm trajectories using Ray for distributed processing.
    """

    def __init__(self, config: IngestionConfig):
        """
        Initialize parallel data ingester.

        Args:
            config: Ingestion configuration
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for parallel ingestion. Install with: pip install 'ray[data]'"
            )

        self.config = config

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**(config.ray_init_kwargs or {}))

        # Create output directory
        os.makedirs(config.output_directory, exist_ok=True)

    def ingest_data(self,
                    ingester: DataIngestionInterface,
                    return_vla_dataset: bool = True) -> List[str]:
        """
        Ingest data using the provided ingester interface.

        Args:
            ingester: Data ingestion interface implementation
            return_vla_dataset: Whether to return a VLADataset object

        Returns:
            List of created trajectory file paths, or VLADataset if return_vla_dataset=True
        """
        logger.info("Starting parallel data ingestion")

        # Get all data items
        logger.info("Discovering data items...")
        all_items = ingester.get_data_items()
        logger.info(f"Found {len(all_items)} data items")

        if not all_items:
            logger.warning("No data items found")
            return []

        # Shuffle if requested
        if self.config.shuffle_items:
            logger.info("Shuffling data items")
            random.shuffle(all_items)

        # Group items into trajectories
        logger.info("Grouping items into trajectories...")
        trajectory_groups = ingester.group_items_into_trajectories(all_items)
        logger.info(f"Created {len(trajectory_groups)} trajectory groups")

        # Split trajectory groups into batches for parallel processing
        batch_size = max(1, len(trajectory_groups) // self.config.num_workers)
        batches = []
        for i in range(0, len(trajectory_groups), batch_size):
            batch = trajectory_groups[i:i + batch_size]
            batches.append(batch)

        logger.info(
            f"Split into {len(batches)} batches for {self.config.num_workers} workers"
        )

        # Create Ray workers
        workers = []
        config_dict = self._config_to_dict()

        for i in range(min(len(batches), self.config.num_workers)):
            worker = TrajectoryWorker.remote(config_dict)

            # Initialize worker with ingester
            ingester_class = type(ingester)
            ingester_kwargs = self._extract_ingester_kwargs(ingester)
            worker.initialize_processor.remote(ingester_class, ingester_kwargs)

            workers.append(worker)

        # Process batches in parallel
        logger.info("Processing trajectory batches in parallel...")
        futures = []

        for i, batch in enumerate(batches):
            worker_idx = i % len(workers)
            future = workers[worker_idx].process_batch.remote(batch)
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        # Flatten results
        all_created_files = []
        for batch_result in results:
            all_created_files.extend(batch_result)

        logger.info(
            f"Successfully created {len(all_created_files)} trajectory files")

        if return_vla_dataset:
            # Import here to avoid circular imports
            from robodm.dataset import DatasetConfig, VLADataset

            # Create dataset config matching ingestion config
            dataset_config = DatasetConfig(
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_items,
                num_parallel_reads=self.config.num_workers,
                ray_init_kwargs=self.config.ray_init_kwargs,
            )

            # Create VLA dataset from the output directory
            return VLADataset.create_trajectory_dataset(
                path=f"{self.config.output_directory}/*.mkv",
                config=dataset_config,
            )

        return all_created_files

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Ray serialization."""
        return {
            "output_directory": self.config.output_directory,
            "trajectory_prefix": self.config.trajectory_prefix,
            "num_workers": self.config.num_workers,
            "batch_size": self.config.batch_size,
            "ray_init_kwargs": self.config.ray_init_kwargs,
            "time_unit": self.config.time_unit,
            "enforce_monotonic": self.config.enforce_monotonic,
            "video_codec": self.config.video_codec,
            "raw_codec": self.config.raw_codec,
            "codec_options": self.config.codec_options,
            "shuffle_items": self.config.shuffle_items,
            "max_items_per_trajectory": self.config.max_items_per_trajectory,
            "metadata": self.config.metadata,
        }

    def _extract_ingester_kwargs(
            self, ingester: DataIngestionInterface) -> Dict[str, Any]:
        """Extract initialization kwargs from ingester instance."""
        # This is a simple implementation - for more complex ingesters,
        # you might need to implement a serialization method

        kwargs = {}

        # Extract common attributes that are typically used for initialization
        for attr in [
                "dataset",
                "transform_fn",
                "group_size",
                "trajectory_name_fn",
                "iterator_factory",
                "max_items",
                "data_generator",
                "file_paths",
        ]:
            if hasattr(ingester, attr):
                kwargs[attr] = getattr(ingester, attr)

        return kwargs


def create_parallel_ingester(output_directory: str,
                             num_workers: int = 4,
                             batch_size: int = 1,
                             **kwargs) -> ParallelDataIngester:
    """
    Create a parallel data ingester with common configuration.

    Args:
        output_directory: Directory where trajectory files will be saved
        num_workers: Number of parallel workers
        batch_size: Batch size for processing
        **kwargs: Additional configuration options

    Returns:
        Configured ParallelDataIngester instance
    """
    config = IngestionConfig(
        output_directory=output_directory,
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs,
    )

    return ParallelDataIngester(config)


@ray.remote
def process_single_trajectory_group(
    trajectory_group: List[Any],
    ingester_class: type,
    ingester_kwargs: Dict[str, Any],
    config_dict: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Ray remote function for processing a single trajectory group.

    This is an alternative to the actor-based approach for simpler use cases.
    """
    # Reconstruct objects
    config = IngestionConfig(**config_dict)
    ingester = ingester_class(**ingester_kwargs)
    processor = BatchProcessor(ingester, config)

    # Process the trajectory group
    result = processor.process_trajectory_groups([trajectory_group])
    return result[0] if result else None


class SimplifiedParallelIngester:
    """
    Simplified version of parallel ingester using Ray remote functions
    instead of actors for lighter use cases.
    """

    def __init__(self, config: IngestionConfig):
        """Initialize simplified parallel ingester."""
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for parallel ingestion. Install with: pip install 'ray[data]'"
            )

        self.config = config

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**(config.ray_init_kwargs or {}))

        # Create output directory
        os.makedirs(config.output_directory, exist_ok=True)

    def ingest_data(self, ingester: DataIngestionInterface) -> List[str]:
        """Ingest data using Ray remote functions."""
        logger.info("Starting simplified parallel data ingestion")

        # Get all data items and group into trajectories
        all_items = ingester.get_data_items()
        trajectory_groups = ingester.group_items_into_trajectories(all_items)

        # Prepare arguments for Ray tasks
        ingester_class = type(ingester)
        ingester_kwargs = self._extract_ingester_kwargs(ingester)
        config_dict = self._config_to_dict()

        # Submit Ray tasks
        futures = []
        for i, group in enumerate(trajectory_groups):
            filename = ingester.get_trajectory_filename(group, i)
            if not filename.endswith(".mkv"):
                filename += ".mkv"
            output_path = str(Path(self.config.output_directory) / filename)

            future = process_single_trajectory_group.remote(
                group, ingester_class, ingester_kwargs, config_dict,
                output_path)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        return [r for r in results if r is not None]

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Ray serialization."""
        return {
            "output_directory": self.config.output_directory,
            "trajectory_prefix": self.config.trajectory_prefix,
            "num_workers": self.config.num_workers,
            "batch_size": self.config.batch_size,
            "ray_init_kwargs": self.config.ray_init_kwargs,
            "time_unit": self.config.time_unit,
            "enforce_monotonic": self.config.enforce_monotonic,
            "video_codec": self.config.video_codec,
            "raw_codec": self.config.raw_codec,
            "codec_options": self.config.codec_options,
            "shuffle_items": self.config.shuffle_items,
            "max_items_per_trajectory": self.config.max_items_per_trajectory,
            "metadata": self.config.metadata,
        }

    def _extract_ingester_kwargs(
            self, ingester: DataIngestionInterface) -> Dict[str, Any]:
        """Extract initialization kwargs from ingester instance."""
        kwargs = {}

        for attr in [
                "dataset",
                "transform_fn",
                "group_size",
                "trajectory_name_fn",
                "iterator_factory",
                "max_items",
                "data_generator",
                "file_paths",
        ]:
            if hasattr(ingester, attr):
                kwargs[attr] = getattr(ingester, attr)

        return kwargs
