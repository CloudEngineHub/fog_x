"""
Base interfaces and configuration for data ingestion into robodm VLA datasets.

This module provides the core abstractions that allow users to define how their
custom data sources should be transformed into robodm trajectories, with automatic
Ray-based parallel processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Text, Union

import numpy as np

from robodm import FeatureType, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion process."""

    # Output configuration
    output_directory: str
    trajectory_prefix: str = "trajectory"

    # Parallel processing
    num_workers: int = 4
    batch_size: int = 1
    ray_init_kwargs: Optional[Dict] = None

    # Trajectory configuration
    time_unit: str = "ms"
    enforce_monotonic: bool = True
    video_codec: str = "auto"
    raw_codec: Optional[str] = None
    codec_options: Optional[Dict[str, Any]] = None

    # Data processing
    shuffle_items: bool = False
    max_items_per_trajectory: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataIngestionInterface(ABC):
    """
    Abstract interface for ingesting data from custom sources into robodm trajectories.

    Users implement this interface to define:
    1. How to discover/enumerate their data items
    2. How to transform each item into trajectory data
    3. Optional metadata and grouping logic
    """

    @abstractmethod
    def get_data_items(self) -> List[Any]:
        """
        Return a list of data items to be processed.

        Each item can be anything (file path, database record, etc.)
        that contains enough information for transform_item() to process.

        Returns:
            List of data items to process
        """
        pass

    @abstractmethod
    def transform_item(self, item: Any) -> Dict[str, Any]:
        """
        Transform a single data item into trajectory data.

        Args:
            item: A single data item from get_data_items()

        Returns:
            Dictionary where:
            - Keys are feature names
            - Values are data to add to trajectory (np.array, images, etc.)

        Example:
            {
                "sensor_reading": np.array([1.0, 2.0, 3.0]),
                "image": rgb_image_array,  # shape (H, W, 3)
                "metadata": {"patient_id": "12345"}
            }
        """
        pass

    def get_item_metadata(self, item: Any) -> Dict[str, Any]:
        """
        Extract metadata for a data item (optional).

        Args:
            item: A single data item from get_data_items()

        Returns:
            Dictionary with metadata about this item
        """
        return {}

    def group_items_into_trajectories(self,
                                      items: List[Any]) -> List[List[Any]]:
        """
        Group data items into trajectories (optional).

        By default, each item becomes its own trajectory.
        Override to group related items (e.g., time series segments).

        Args:
            items: List of all data items

        Returns:
            List of lists, where each inner list contains items for one trajectory
        """
        return [[item] for item in items]

    def get_trajectory_filename(self, trajectory_group: List[Any],
                                index: int) -> str:
        """
        Generate filename for a trajectory (optional).

        Args:
            trajectory_group: List of items that will form this trajectory
            index: Index of this trajectory in the overall list

        Returns:
            Filename for the trajectory (without extension)
        """
        return f"trajectory_{index:06d}"

    def validate_transformed_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate transformed data before adding to trajectory (optional).

        Args:
            data: Dictionary returned by transform_item()

        Returns:
            True if data is valid, False to skip this item
        """
        return True


class TrajectoryBuilder:
    """Helper class for building trajectories from ingested data."""

    def __init__(self, config: IngestionConfig):
        self.config = config

    def create_trajectory_from_group(
        self,
        trajectory_group: List[Any],
        ingester: DataIngestionInterface,
        output_path: str,
    ) -> str:
        """
        Create a single trajectory file from a group of data items.

        Args:
            trajectory_group: List of items to include in this trajectory
            ingester: Data ingestion interface for transforming items
            output_path: Full path where trajectory should be saved

        Returns:
            Path to created trajectory file
        """
        trajectory = Trajectory(
            output_path,
            mode="w",
            time_unit=self.config.time_unit,
            enforce_monotonic=self.config.enforce_monotonic,
            video_codec=self.config.video_codec,
            raw_codec=self.config.raw_codec,
            codec_options=self.config.codec_options,
        )

        current_timestamp = 0
        items_added = 0

        try:
            for item in trajectory_group:
                # Transform the item
                try:
                    transformed_data = ingester.transform_item(item)
                except Exception as e:
                    logger.warning(f"Failed to transform item {item}: {e}")
                    continue

                # Validate the transformed data
                if not ingester.validate_transformed_data(transformed_data):
                    logger.debug(f"Skipping invalid data for item {item}")
                    continue

                # Add to trajectory
                trajectory.add_by_dict(
                    transformed_data,
                    timestamp=current_timestamp,
                    time_unit=self.config.time_unit,
                )

                current_timestamp += 100  # 100ms intervals by default
                items_added += 1

                # Check max items limit
                if (self.config.max_items_per_trajectory and items_added
                        >= self.config.max_items_per_trajectory):
                    break

        finally:
            trajectory.close()

        logger.info(
            f"Created trajectory {output_path} with {items_added} items")
        return output_path


class BatchProcessor:
    """Helper for processing data items in batches."""

    def __init__(self, ingester: DataIngestionInterface,
                 config: IngestionConfig):
        self.ingester = ingester
        self.config = config
        self.builder = TrajectoryBuilder(config)

    def process_trajectory_groups(
            self, trajectory_groups: List[List[Any]]) -> List[str]:
        """
        Process multiple trajectory groups and return created file paths.

        Args:
            trajectory_groups: List of trajectory groups to process

        Returns:
            List of created trajectory file paths
        """
        created_files = []

        for i, group in enumerate(trajectory_groups):
            # Generate filename
            filename = self.ingester.get_trajectory_filename(group, i)
            if not filename.endswith(".mkv"):
                filename += ".mkv"

            output_path = str(Path(self.config.output_directory) / filename)

            try:
                created_path = self.builder.create_trajectory_from_group(
                    group, self.ingester, output_path)
                created_files.append(created_path)
            except Exception as e:
                logger.error(f"Failed to create trajectory {output_path}: {e}")

        return created_files
