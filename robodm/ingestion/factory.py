"""
Factory functions for creating VLA datasets from various data sources.

This module provides high-level convenience functions that users can call
to quickly create VLA datasets from their data with minimal code changes.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .adapters import (CallableAdapter, FileListAdapter, IteratorAdapter,
                       PyTorchDatasetAdapter)
from .base import DataIngestionInterface, IngestionConfig
from .parallel import ParallelDataIngester

logger = logging.getLogger(__name__)


def create_vla_dataset_from_source(
    data_source: Union[Any, Iterator, Callable, List[str],
                       DataIngestionInterface],
    output_directory: Optional[str] = None,
    transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    group_size: int = 1,
    num_workers: int = 4,
    return_vla_dataset: bool = True,
    **kwargs,
):
    """
    Create a VLA dataset from various data sources with automatic adaptation.

    This is the main factory function that users should call to create VLA datasets
    from their existing data sources with minimal code changes.

    Args:
        data_source: Can be:
            - PyTorch Dataset object (with __len__ and __getitem__)
            - Iterator factory function (returns Iterator)
            - Callable function (returns List[Any])
            - List of file paths
            - Custom DataIngestionInterface implementation
        output_directory: Directory to save trajectory files (temp dir if None)
        transform_fn: Function to transform data items into robodm format
        group_size: Number of data items to group into each trajectory
        num_workers: Number of parallel workers for processing
        return_vla_dataset: If True, return VLADataset; if False, return file paths
        **kwargs: Additional configuration options

    Returns:
        VLADataset object or list of trajectory file paths

    Examples:
        # From PyTorch dataset
        >>> pytorch_dataset = MyPyTorchDataset()
        >>> vla_dataset = create_vla_dataset_from_source(
        ...     pytorch_dataset,
        ...     transform_fn=lambda x: {"image": x[0], "label": x[1]}
        ... )

        # From file list
        >>> file_paths = ["data1.json", "data2.json", "data3.json"]
        >>> vla_dataset = create_vla_dataset_from_source(
        ...     file_paths,
        ...     transform_fn=lambda path: load_and_transform(path)
        ... )

        # From iterator
        >>> def data_iterator():
        ...     for i in range(1000):
        ...         yield generate_data_item(i)
        >>> vla_dataset = create_vla_dataset_from_source(
        ...     data_iterator,
        ...     transform_fn=lambda item: {"sensor_data": item}
        ... )
    """
    # Create output directory if not provided
    if output_directory is None:
        output_directory = tempfile.mkdtemp(prefix="robodm_trajectories_")
        logger.info(f"Using temporary directory: {output_directory}")

    # Create ingestion config
    config = IngestionConfig(output_directory=output_directory,
                             num_workers=num_workers,
                             **kwargs)

    # Automatically adapt the data source
    ingester = _auto_adapt_data_source(data_source=data_source,
                                       transform_fn=transform_fn,
                                       group_size=group_size)

    # Create parallel ingester and process data
    parallel_ingester = ParallelDataIngester(config)
    result = parallel_ingester.ingest_data(
        ingester=ingester, return_vla_dataset=return_vla_dataset)

    return result


def create_vla_dataset_from_pytorch_dataset(
    dataset: Any,  # torch.utils.data.Dataset
    output_directory: Optional[str] = None,
    transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    trajectories_per_dataset: int = 1,
    num_workers: int = 4,
    **kwargs,
):
    """
    Create VLA dataset from PyTorch Dataset with sensible defaults.

    Args:
        dataset: PyTorch dataset object
        output_directory: Directory to save trajectories
        transform_fn: Function to transform dataset items
        trajectories_per_dataset: Number of trajectories to split dataset into
        num_workers: Number of parallel workers
        **kwargs: Additional configuration options

    Returns:
        VLADataset object
    """
    # Calculate group size to get desired number of trajectories
    group_size = max(1, len(dataset) // trajectories_per_dataset)

    return create_vla_dataset_from_source(
        data_source=dataset,
        output_directory=output_directory,
        transform_fn=transform_fn,
        group_size=group_size,
        num_workers=num_workers,
        **kwargs,
    )


def create_vla_dataset_from_file_list(
    file_paths: List[str],
    transform_fn: Callable[[str], Dict[str, Any]],
    output_directory: Optional[str] = None,
    files_per_trajectory: int = 100,
    num_workers: int = 4,
    **kwargs,
):
    """
    Create VLA dataset from list of file paths.

    Args:
        file_paths: List of file paths to process
        transform_fn: Function to transform file path into trajectory data
        output_directory: Directory to save trajectories
        files_per_trajectory: Number of files to include in each trajectory
        num_workers: Number of parallel workers
        **kwargs: Additional configuration options

    Returns:
        VLADataset object
    """
    return create_vla_dataset_from_source(
        data_source=file_paths,
        output_directory=output_directory,
        transform_fn=transform_fn,
        group_size=files_per_trajectory,
        num_workers=num_workers,
        **kwargs,
    )


def create_vla_dataset_from_iterator(
    iterator_factory: Callable[[], Iterator[Any]],
    transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    output_directory: Optional[str] = None,
    max_items: Optional[int] = None,
    items_per_trajectory: int = 100,
    num_workers: int = 4,
    **kwargs,
):
    """
    Create VLA dataset from iterator or generator function.

    Args:
        iterator_factory: Function that returns an iterator
        transform_fn: Function to transform iterator items
        output_directory: Directory to save trajectories
        max_items: Maximum number of items to consume from iterator
        items_per_trajectory: Number of items to include in each trajectory
        num_workers: Number of parallel workers
        **kwargs: Additional configuration options

    Returns:
        VLADataset object
    """
    adapter = IteratorAdapter(
        iterator_factory=iterator_factory,
        transform_fn=transform_fn,
        group_size=items_per_trajectory,
        max_items=max_items,
    )

    config = IngestionConfig(
        output_directory=output_directory
        or tempfile.mkdtemp(prefix="robodm_trajectories_"),
        num_workers=num_workers,
        **kwargs,
    )

    parallel_ingester = ParallelDataIngester(config)
    return parallel_ingester.ingest_data(ingester=adapter,
                                         return_vla_dataset=True)


def create_vla_dataset_from_callable(
    data_generator: Callable[[], List[Any]],
    transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    output_directory: Optional[str] = None,
    items_per_trajectory: int = 100,
    num_workers: int = 4,
    **kwargs,
):
    """
    Create VLA dataset from callable that generates data.

    Args:
        data_generator: Function that returns list of data items
        transform_fn: Function to transform generated items
        output_directory: Directory to save trajectories
        items_per_trajectory: Number of items to include in each trajectory
        num_workers: Number of parallel workers
        **kwargs: Additional configuration options

    Returns:
        VLADataset object
    """
    adapter = CallableAdapter(
        data_generator=data_generator,
        transform_fn=transform_fn,
        group_size=items_per_trajectory,
    )

    config = IngestionConfig(
        output_directory=output_directory
        or tempfile.mkdtemp(prefix="robodm_trajectories_"),
        num_workers=num_workers,
        **kwargs,
    )

    parallel_ingester = ParallelDataIngester(config)
    return parallel_ingester.ingest_data(ingester=adapter,
                                         return_vla_dataset=True)


def _auto_adapt_data_source(
    data_source: Union[Any, Iterator, Callable, List[str],
                       DataIngestionInterface],
    transform_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    group_size: int = 1,
) -> DataIngestionInterface:
    """
    Automatically adapt a data source to the DataIngestionInterface.

    Args:
        data_source: The data source to adapt
        transform_fn: Optional transformation function
        group_size: Number of items per trajectory group

    Returns:
        DataIngestionInterface implementation
    """
    # If already an ingester, return as-is
    if isinstance(data_source, DataIngestionInterface):
        return data_source

    # Check if it's a PyTorch dataset (has __len__ and __getitem__)
    if hasattr(data_source, "__len__") and hasattr(data_source, "__getitem__"):
        logger.info("Detected PyTorch-style dataset")
        return PyTorchDatasetAdapter(
            dataset=data_source,
            transform_fn=transform_fn,
            group_size=group_size,
        )

    # Check if it's a list of strings (file paths)
    if isinstance(data_source, list) and all(
            isinstance(x, str) for x in data_source):
        logger.info("Detected file list")
        if transform_fn is None:
            raise ValueError(
                "transform_fn is required for file list data sources")
        return FileListAdapter(
            file_paths=data_source,
            transform_fn=transform_fn,
            group_size=group_size,
        )

    # Check if it's a callable that returns an iterator
    if callable(data_source):
        try:
            # Try calling it to see what it returns
            result = data_source()
            if hasattr(result,
                       "__iter__") and not isinstance(result, (str, bytes)):
                logger.info("Detected iterator factory")
                return IteratorAdapter(
                    iterator_factory=data_source,
                    transform_fn=transform_fn,
                    group_size=group_size,
                )
            elif isinstance(result, list):
                logger.info("Detected callable data generator")
                return CallableAdapter(
                    data_generator=data_source,
                    transform_fn=transform_fn,
                    group_size=group_size,
                )
        except Exception as e:
            logger.warning(f"Failed to auto-detect callable type: {e}")

    # Check if it's an iterator directly
    if hasattr(data_source,
               "__iter__") and not isinstance(data_source, (str, bytes, list)):
        logger.info("Detected iterator")
        # Wrap in a factory function
        items = list(data_source)  # Consume the iterator
        return CallableAdapter(
            data_generator=lambda: items,
            transform_fn=transform_fn,
            group_size=group_size,
        )

    raise ValueError(
        f"Unable to auto-adapt data source of type {type(data_source)}. "
        f"Please provide a custom DataIngestionInterface implementation or use one of the "
        f"supported types: PyTorch Dataset, Iterator, Callable, List[str], or DataIngestionInterface."
    )
