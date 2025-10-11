"""
Adapter classes for wrapping existing data sources into the ingestion interface.

These adapters allow users to quickly integrate existing PyTorch datasets,
iterators, or callable functions with the robodm ingestion system.
"""

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .base import DataIngestionInterface

logger = logging.getLogger(__name__)


class PyTorchDatasetAdapter(DataIngestionInterface):
    """
    Adapter for PyTorch Dataset objects.

    This allows users to directly use existing PyTorch datasets with the
    robodm ingestion system.
    """

    def __init__(
        self,
        dataset: Any,  # torch.utils.data.Dataset
        transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        group_size: int = 1,
        trajectory_name_fn: Optional[Callable[[List[Any], int], str]] = None,
    ):
        """
        Initialize PyTorch dataset adapter.

        Args:
            dataset: PyTorch dataset object with __len__ and __getitem__
            transform_fn: Optional function to transform dataset items into robodm format
                        If None, assumes dataset items are already dicts with proper format
            group_size: Number of dataset items to group into each trajectory
            trajectory_name_fn: Optional function to generate trajectory names
        """
        self.dataset = dataset
        self.transform_fn = transform_fn
        self.group_size = group_size
        self.trajectory_name_fn = trajectory_name_fn

        # Validate dataset interface
        if not hasattr(dataset, "__len__") or not hasattr(
                dataset, "__getitem__"):
            raise ValueError("Dataset must implement __len__ and __getitem__")

    def get_data_items(self) -> List[Any]:
        """Return indices into the PyTorch dataset."""
        return list(range(len(self.dataset)))

    def transform_item(self, item: Any) -> Dict[str, Any]:
        """Transform a dataset index into trajectory data."""
        # Get the actual data from the dataset
        data = self.dataset[item]

        # Apply transformation if provided
        if self.transform_fn:
            return self.transform_fn(data)

        # Assume data is already in correct format
        if isinstance(data, dict):
            return data
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            # Common PyTorch pattern: (input, label)
            return {"input": data[0], "label": data[1]}
        else:
            # Single item - use generic name
            return {"data": data}

    def group_items_into_trajectories(self,
                                      items: List[Any]) -> List[List[Any]]:
        """Group dataset indices into trajectory groups."""
        groups = []
        for i in range(0, len(items), self.group_size):
            group = items[i:i + self.group_size]
            groups.append(group)
        return groups

    def get_trajectory_filename(self, trajectory_group: List[Any],
                                index: int) -> str:
        """Generate trajectory filename."""
        if self.trajectory_name_fn:
            return self.trajectory_name_fn(trajectory_group, index)

        start_idx = trajectory_group[0]
        end_idx = trajectory_group[-1]
        return f"pytorch_dataset_trajectory_{start_idx:06d}_{end_idx:06d}"


class IteratorAdapter(DataIngestionInterface):
    """
    Adapter for iterator objects or generator functions.

    This allows users to wrap existing iterators or generators to work
    with the robodm ingestion system.
    """

    def __init__(
        self,
        iterator_factory: Callable[[], Iterator[Any]],
        transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        group_size: int = 1,
        max_items: Optional[int] = None,
        trajectory_name_fn: Optional[Callable[[List[Any], int], str]] = None,
    ):
        """
        Initialize iterator adapter.

        Args:
            iterator_factory: Function that returns a new iterator instance
            transform_fn: Optional function to transform iterator items into robodm format
            group_size: Number of iterator items to group into each trajectory
            max_items: Maximum number of items to consume from iterator
            trajectory_name_fn: Optional function to generate trajectory names
        """
        self.iterator_factory = iterator_factory
        self.transform_fn = transform_fn
        self.group_size = group_size
        self.max_items = max_items
        self.trajectory_name_fn = trajectory_name_fn
        self._cached_items: Optional[List[Any]] = None

    def get_data_items(self) -> List[Any]:
        """Consume iterator and cache items."""
        if self._cached_items is None:
            self._cached_items = []
            iterator = self.iterator_factory()

            for i, item in enumerate(iterator):
                if self.max_items and i >= self.max_items:
                    break
                self._cached_items.append(item)

        return self._cached_items

    def transform_item(self, item: Any) -> Dict[str, Any]:
        """Transform an iterator item into trajectory data."""
        if self.transform_fn:
            return self.transform_fn(item)

        # Assume item is already in correct format
        if isinstance(item, dict):
            return item
        else:
            return {"data": item}

    def group_items_into_trajectories(self,
                                      items: List[Any]) -> List[List[Any]]:
        """Group iterator items into trajectory groups."""
        groups = []
        for i in range(0, len(items), self.group_size):
            group = items[i:i + self.group_size]
            groups.append(group)
        return groups

    def get_trajectory_filename(self, trajectory_group: List[Any],
                                index: int) -> str:
        """Generate trajectory filename."""
        if self.trajectory_name_fn:
            return self.trajectory_name_fn(trajectory_group, index)

        return f"iterator_trajectory_{index:06d}"


class CallableAdapter(DataIngestionInterface):
    """
    Adapter for callable functions that generate data.

    This allows users to wrap functions that generate data items
    to work with the robodm ingestion system.
    """

    def __init__(
        self,
        data_generator: Callable[[], List[Any]],
        transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        group_size: int = 1,
        trajectory_name_fn: Optional[Callable[[List[Any], int], str]] = None,
    ):
        """
        Initialize callable adapter.

        Args:
            data_generator: Function that returns a list of data items
            transform_fn: Optional function to transform items into robodm format
            group_size: Number of items to group into each trajectory
            trajectory_name_fn: Optional function to generate trajectory names
        """
        self.data_generator = data_generator
        self.transform_fn = transform_fn
        self.group_size = group_size
        self.trajectory_name_fn = trajectory_name_fn

    def get_data_items(self) -> List[Any]:
        """Generate data items using the callable."""
        return self.data_generator()

    def transform_item(self, item: Any) -> Dict[str, Any]:
        """Transform a generated item into trajectory data."""
        if self.transform_fn:
            return self.transform_fn(item)

        # Assume item is already in correct format
        if isinstance(item, dict):
            return item
        else:
            return {"data": item}

    def group_items_into_trajectories(self,
                                      items: List[Any]) -> List[List[Any]]:
        """Group generated items into trajectory groups."""
        groups = []
        for i in range(0, len(items), self.group_size):
            group = items[i:i + self.group_size]
            groups.append(group)
        return groups

    def get_trajectory_filename(self, trajectory_group: List[Any],
                                index: int) -> str:
        """Generate trajectory filename."""
        if self.trajectory_name_fn:
            return self.trajectory_name_fn(trajectory_group, index)

        return f"callable_trajectory_{index:06d}"


class FileListAdapter(DataIngestionInterface):
    """
    Adapter for file lists with a transformation function.

    This is useful for processing directories of files, database exports, etc.
    """

    def __init__(
        self,
        file_paths: List[str],
        transform_fn: Callable[[str], Dict[str, Any]],
        group_size: int = 1,
        trajectory_name_fn: Optional[Callable[[List[Any], int], str]] = None,
    ):
        """
        Initialize file list adapter.

        Args:
            file_paths: List of file paths to process
            transform_fn: Function to transform file path into robodm format
            group_size: Number of files to group into each trajectory
            trajectory_name_fn: Optional function to generate trajectory names
        """
        self.file_paths = file_paths
        self.transform_fn = transform_fn
        self.group_size = group_size
        self.trajectory_name_fn = trajectory_name_fn

    def get_data_items(self) -> List[Any]:
        """Return the list of file paths."""
        return self.file_paths

    def transform_item(self, item: Any) -> Dict[str, Any]:
        """Transform a file path into trajectory data."""
        return self.transform_fn(item)

    def group_items_into_trajectories(self,
                                      items: List[Any]) -> List[List[Any]]:
        """Group file paths into trajectory groups."""
        groups = []
        for i in range(0, len(items), self.group_size):
            group = items[i:i + self.group_size]
            groups.append(group)
        return groups

    def get_trajectory_filename(self, trajectory_group: List[Any],
                                index: int) -> str:
        """Generate trajectory filename."""
        if self.trajectory_name_fn:
            return self.trajectory_name_fn(trajectory_group, index)

        # Use first file's name as base
        first_file = trajectory_group[0]
        base_name = str(first_file).split("/")[-1].split(".")[0]
        return f"file_trajectory_{base_name}_{index:06d}"
