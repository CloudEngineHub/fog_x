"""Tests for the data ingestion system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from robodm.ingestion.adapters import (CallableAdapter, FileListAdapter,
                                       IteratorAdapter, PyTorchDatasetAdapter)
from robodm.ingestion.base import (BatchProcessor, DataIngestionInterface,
                                   IngestionConfig, TrajectoryBuilder)
from robodm.ingestion.factory import (_auto_adapt_data_source,
                                      create_vla_dataset_from_callable,
                                      create_vla_dataset_from_file_list,
                                      create_vla_dataset_from_iterator,
                                      create_vla_dataset_from_pytorch_dataset,
                                      create_vla_dataset_from_source)

if RAY_AVAILABLE:
    from robodm.ingestion.parallel import ParallelDataIngester


class MockPyTorchDataset:
    """Mock PyTorch dataset for testing."""

    def __init__(self, size=10):
        self.size = size
        self.data = [{
            "input": np.random.rand(3, 32, 32),
            "label": i % 2
        } for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MockDataIngester(DataIngestionInterface):
    """Mock data ingester for testing."""

    def __init__(self, items=None):
        self.items = items or [f"item_{i}" for i in range(5)]

    def get_data_items(self):
        return self.items

    def transform_item(self, item):
        return {"data": f"transformed_{item}", "value": np.random.rand(3)}

    def get_trajectory_filename(self, trajectory_group, index):
        return f"test_trajectory_{index}"


@pytest.fixture
def sample_config(temp_dir):
    """Create sample ingestion config."""
    return IngestionConfig(output_directory=str(temp_dir),
                           num_workers=2,
                           time_unit="ms")


@pytest.fixture
def mock_trajectory():
    """Mock Trajectory object."""
    with patch("robodm.ingestion.base.Trajectory") as mock_traj_class:
        mock_traj = Mock()
        mock_traj_class.return_value = mock_traj
        yield mock_traj


class TestIngestionConfig:
    """Test IngestionConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IngestionConfig(output_directory="/tmp")

        assert config.output_directory == "/tmp"
        assert config.trajectory_prefix == "trajectory"
        assert config.num_workers == 4
        assert config.batch_size == 1
        assert config.time_unit == "ms"
        assert config.enforce_monotonic is True
        assert config.video_codec == "auto"
        assert config.shuffle_items is False
        assert config.metadata == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        metadata = {"experiment": "test"}
        config = IngestionConfig(
            output_directory="/custom",
            num_workers=8,
            batch_size=32,
            video_codec="libx264",
            metadata=metadata,
        )

        assert config.output_directory == "/custom"
        assert config.num_workers == 8
        assert config.batch_size == 32
        assert config.video_codec == "libx264"
        assert config.metadata == metadata


class TestTrajectoryBuilder:
    """Test TrajectoryBuilder class."""

    def test_create_trajectory_from_group(self, sample_config, mock_trajectory,
                                          temp_dir):
        """Test creating trajectory from group of items."""
        builder = TrajectoryBuilder(sample_config)
        ingester = MockDataIngester(["item1", "item2"])
        output_path = str(temp_dir / "test_trajectory.mkv")

        result = builder.create_trajectory_from_group(["item1", "item2"],
                                                      ingester, output_path)

        assert result == output_path
        mock_trajectory.add_by_dict.assert_has_calls([
            call(
                {
                    "data":
                    "transformed_item1",
                    "value":
                    mock_trajectory.add_by_dict.call_args_list[0][0][0]
                    ["value"],
                },
                timestamp=0,
                time_unit="ms",
            ),
            call(
                {
                    "data":
                    "transformed_item2",
                    "value":
                    mock_trajectory.add_by_dict.call_args_list[1][0][0]
                    ["value"],
                },
                timestamp=100,
                time_unit="ms",
            ),
        ])
        mock_trajectory.close.assert_called_once()

    def test_create_trajectory_with_transform_error(self, sample_config,
                                                    mock_trajectory, temp_dir):
        """Test handling of transform errors."""
        builder = TrajectoryBuilder(sample_config)

        # Create ingester that fails on second item
        ingester = MockDataIngester()
        original_transform = ingester.transform_item

        def failing_transform(item):
            if item == "item2":
                raise ValueError("Transform failed")
            return original_transform(item)

        ingester.transform_item = failing_transform

        output_path = str(temp_dir / "test_trajectory.mkv")

        with patch("robodm.ingestion.base.logger") as mock_logger:
            result = builder.create_trajectory_from_group(
                ["item1", "item2", "item3"], ingester, output_path)

        assert result == output_path
        assert mock_trajectory.add_by_dict.call_count == 2  # item2 skipped
        mock_logger.warning.assert_called_once()

    def test_create_trajectory_with_max_items(self, sample_config,
                                              mock_trajectory, temp_dir):
        """Test max items per trajectory limit."""
        sample_config.max_items_per_trajectory = 2
        builder = TrajectoryBuilder(sample_config)
        ingester = MockDataIngester()
        output_path = str(temp_dir / "test_trajectory.mkv")

        result = builder.create_trajectory_from_group(
            ["item1", "item2", "item3", "item4"], ingester, output_path)

        assert result == output_path
        assert mock_trajectory.add_by_dict.call_count == 2  # Limited to 2 items


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_process_trajectory_groups(self, sample_config, mock_trajectory,
                                       temp_dir):
        """Test processing multiple trajectory groups."""
        ingester = MockDataIngester()
        processor = BatchProcessor(ingester, sample_config)

        trajectory_groups = [["item1", "item2"], ["item3", "item4"]]

        with patch.object(processor.builder,
                          "create_trajectory_from_group") as mock_create:
            mock_create.side_effect = [
                str(temp_dir / "test_trajectory_0.mkv"),
                str(temp_dir / "test_trajectory_1.mkv"),
            ]

            result = processor.process_trajectory_groups(trajectory_groups)

        assert len(result) == 2
        assert mock_create.call_count == 2

        # Check filenames were generated correctly
        call_args = mock_create.call_args_list
        assert "test_trajectory_0.mkv" in call_args[0][0][2]
        assert "test_trajectory_1.mkv" in call_args[1][0][2]

    def test_process_trajectory_groups_with_errors(self, sample_config,
                                                   temp_dir):
        """Test handling errors during trajectory creation."""
        ingester = MockDataIngester()
        processor = BatchProcessor(ingester, sample_config)

        trajectory_groups = [["item1"], ["item2"]]

        with patch.object(processor.builder,
                          "create_trajectory_from_group") as mock_create:
            mock_create.side_effect = [
                str(temp_dir / "success.mkv"),
                Exception("Creation failed"),
            ]

            with patch("robodm.ingestion.base.logger") as mock_logger:
                result = processor.process_trajectory_groups(trajectory_groups)

        assert len(result) == 1  # Only successful trajectory
        mock_logger.error.assert_called_once()


class TestPyTorchDatasetAdapter:
    """Test PyTorchDatasetAdapter class."""

    def test_init_valid_dataset(self):
        """Test initialization with valid PyTorch dataset."""
        dataset = MockPyTorchDataset(5)
        adapter = PyTorchDatasetAdapter(dataset, group_size=2)

        assert adapter.dataset == dataset
        assert adapter.group_size == 2
        assert adapter.transform_fn is None

    def test_init_invalid_dataset(self):
        """Test initialization with invalid dataset."""
        invalid_dataset = "not a dataset"

        with pytest.raises(ValueError,
                           match="must implement __len__ and __getitem__"):
            PyTorchDatasetAdapter(invalid_dataset)

    def test_get_data_items(self):
        """Test getting data items (indices)."""
        dataset = MockPyTorchDataset(5)
        adapter = PyTorchDatasetAdapter(dataset)

        items = adapter.get_data_items()
        assert items == [0, 1, 2, 3, 4]

    def test_transform_item_without_transform_fn(self):
        """Test transforming item without custom transform function."""
        dataset = MockPyTorchDataset(3)
        adapter = PyTorchDatasetAdapter(dataset)

        result = adapter.transform_item(0)

        assert "input" in result
        assert "label" in result
        assert result["label"] == 0

    def test_transform_item_with_transform_fn(self):
        """Test transforming item with custom transform function."""
        dataset = MockPyTorchDataset(3)

        def custom_transform(data):
            return {"image": data["input"], "class": data["label"]}

        adapter = PyTorchDatasetAdapter(dataset, transform_fn=custom_transform)
        result = adapter.transform_item(0)

        assert "image" in result
        assert "class" in result
        assert result["class"] == 0

    def test_transform_item_single_value(self):
        """Test transforming single value items."""

        class SimpleDataset:

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return np.array([idx, idx + 1])

        adapter = PyTorchDatasetAdapter(SimpleDataset())
        result = adapter.transform_item(1)

        assert "data" in result
        assert np.array_equal(result["data"], np.array([1, 2]))

    def test_group_items_into_trajectories(self):
        """Test grouping items into trajectories."""
        dataset = MockPyTorchDataset(7)
        adapter = PyTorchDatasetAdapter(dataset, group_size=3)

        items = adapter.get_data_items()
        groups = adapter.group_items_into_trajectories(items)

        assert len(groups) == 3  # 7 items / 3 = 2 full groups + 1 partial
        assert groups[0] == [0, 1, 2]
        assert groups[1] == [3, 4, 5]
        assert groups[2] == [6]

    def test_get_trajectory_filename(self):
        """Test trajectory filename generation."""
        dataset = MockPyTorchDataset(5)
        adapter = PyTorchDatasetAdapter(dataset)

        filename = adapter.get_trajectory_filename([0, 1, 2], 0)
        assert filename == "pytorch_dataset_trajectory_000000_000002"

    def test_get_trajectory_filename_custom(self):
        """Test custom trajectory filename generation."""
        dataset = MockPyTorchDataset(5)

        def custom_name_fn(group, index):
            return f"custom_{index}_{len(group)}"

        adapter = PyTorchDatasetAdapter(dataset,
                                        trajectory_name_fn=custom_name_fn)
        filename = adapter.get_trajectory_filename([0, 1], 5)

        assert filename == "custom_5_2"


class TestIteratorAdapter:
    """Test IteratorAdapter class."""

    def test_init(self):
        """Test initialization."""

        def iterator_factory():
            return iter([1, 2, 3])

        adapter = IteratorAdapter(iterator_factory, group_size=2)

        assert adapter.iterator_factory == iterator_factory
        assert adapter.group_size == 2
        assert adapter._cached_items is None

    def test_get_data_items(self):
        """Test getting data items from iterator."""

        def iterator_factory():
            return iter(["a", "b", "c", "d"])

        adapter = IteratorAdapter(iterator_factory)
        items = adapter.get_data_items()

        assert items == ["a", "b", "c", "d"]
        assert adapter._cached_items == items

        # Second call should use cache
        items2 = adapter.get_data_items()
        assert items2 is items

    def test_get_data_items_with_max_items(self):
        """Test getting data items with max_items limit."""

        def iterator_factory():
            return iter(range(10))

        adapter = IteratorAdapter(iterator_factory, max_items=5)
        items = adapter.get_data_items()

        assert items == [0, 1, 2, 3, 4]

    def test_transform_item_without_transform_fn(self):
        """Test transforming item without custom transform function."""

        def iterator_factory():
            return iter([{"key": "value"}])

        adapter = IteratorAdapter(iterator_factory)
        result = adapter.transform_item({"key": "value"})

        assert result == {"key": "value"}

    def test_transform_item_with_transform_fn(self):
        """Test transforming item with custom transform function."""

        def iterator_factory():
            return iter([1, 2, 3])

        def transform_fn(item):
            return {"number": item, "squared": item**2}

        adapter = IteratorAdapter(iterator_factory, transform_fn=transform_fn)
        result = adapter.transform_item(3)

        assert result == {"number": 3, "squared": 9}

    def test_transform_item_fallback(self):
        """Test transforming non-dict item."""

        def iterator_factory():
            return iter([42])

        adapter = IteratorAdapter(iterator_factory)
        result = adapter.transform_item(42)

        assert result == {"data": 42}

    def test_group_items_into_trajectories(self):
        """Test grouping iterator items."""

        def iterator_factory():
            return iter(range(5))

        adapter = IteratorAdapter(iterator_factory, group_size=2)
        items = adapter.get_data_items()
        groups = adapter.group_items_into_trajectories(items)

        assert groups == [[0, 1], [2, 3], [4]]

    def test_get_trajectory_filename(self):
        """Test trajectory filename generation."""

        def iterator_factory():
            return iter([])

        adapter = IteratorAdapter(iterator_factory)
        filename = adapter.get_trajectory_filename([], 3)

        assert filename == "iterator_trajectory_000003"


class TestCallableAdapter:
    """Test CallableAdapter class."""

    def test_init(self):
        """Test initialization."""

        def data_generator():
            return [1, 2, 3]

        adapter = CallableAdapter(data_generator, group_size=2)

        assert adapter.data_generator == data_generator
        assert adapter.group_size == 2

    def test_get_data_items(self):
        """Test getting data items from callable."""

        def data_generator():
            return ["x", "y", "z"]

        adapter = CallableAdapter(data_generator)
        items = adapter.get_data_items()

        assert items == ["x", "y", "z"]

    def test_transform_item(self):
        """Test transforming items."""

        def data_generator():
            return [1, 2, 3]

        def transform_fn(item):
            return {"value": item * 10}

        adapter = CallableAdapter(data_generator, transform_fn=transform_fn)
        result = adapter.transform_item(2)

        assert result == {"value": 20}

    def test_get_trajectory_filename(self):
        """Test trajectory filename generation."""

        def data_generator():
            return []

        adapter = CallableAdapter(data_generator)
        filename = adapter.get_trajectory_filename([], 7)

        assert filename == "callable_trajectory_000007"


class TestFileListAdapter:
    """Test FileListAdapter class."""

    def test_init(self):
        """Test initialization."""
        file_paths = ["file1.txt", "file2.txt"]

        def transform_fn(path):
            return {"filename": path}

        adapter = FileListAdapter(file_paths, transform_fn, group_size=1)

        assert adapter.file_paths == file_paths
        assert adapter.transform_fn == transform_fn
        assert adapter.group_size == 1

    def test_get_data_items(self):
        """Test getting file paths."""
        file_paths = ["a.txt", "b.txt", "c.txt"]

        def transform_fn(path):
            return {"file": path}

        adapter = FileListAdapter(file_paths, transform_fn)
        items = adapter.get_data_items()

        assert items == file_paths

    def test_transform_item(self):
        """Test transforming file paths."""

        def transform_fn(path):
            return {"filepath": path, "size": len(path)}

        adapter = FileListAdapter([], transform_fn)
        result = adapter.transform_item("test.txt")

        assert result == {"filepath": "test.txt", "size": 8}

    def test_get_trajectory_filename(self):
        """Test trajectory filename generation from file paths."""

        def transform_fn(path):
            return {}

        adapter = FileListAdapter([], transform_fn)
        filename = adapter.get_trajectory_filename(["/path/to/data.json"], 2)

        assert filename == "file_trajectory_data_000002"


class TestFactoryFunctions:
    """Test factory functions."""

    def test_auto_adapt_pytorch_dataset(self):
        """Test auto-adapting PyTorch dataset."""
        dataset = MockPyTorchDataset(5)

        adapter = _auto_adapt_data_source(dataset)

        assert isinstance(adapter, PyTorchDatasetAdapter)
        assert adapter.dataset == dataset

    def test_auto_adapt_file_list(self):
        """Test auto-adapting file list."""
        file_paths = ["file1.txt", "file2.txt"]

        def transform_fn(path):
            return {"file": path}

        adapter = _auto_adapt_data_source(file_paths, transform_fn)

        assert isinstance(adapter, FileListAdapter)
        assert adapter.file_paths == file_paths

    def test_auto_adapt_file_list_no_transform(self):
        """Test auto-adapting file list without transform function."""
        file_paths = ["file1.txt", "file2.txt"]

        with pytest.raises(ValueError, match="transform_fn is required"):
            _auto_adapt_data_source(file_paths)

    def test_auto_adapt_callable_iterator(self):
        """Test auto-adapting callable that returns iterator."""

        def iterator_factory():
            return iter([1, 2, 3])

        adapter = _auto_adapt_data_source(iterator_factory)

        assert isinstance(adapter, IteratorAdapter)
        assert adapter.iterator_factory == iterator_factory

    def test_auto_adapt_callable_list(self):
        """Test auto-adapting callable that returns list."""

        def data_generator():
            return [1, 2, 3]

        adapter = _auto_adapt_data_source(data_generator)

        assert isinstance(adapter, CallableAdapter)
        assert adapter.data_generator == data_generator

    def test_auto_adapt_existing_interface(self):
        """Test auto-adapting existing DataIngestionInterface."""
        existing_ingester = MockDataIngester()

        adapter = _auto_adapt_data_source(existing_ingester)

        assert adapter is existing_ingester

    def test_auto_adapt_direct_iterator(self):
        """Test auto-adapting direct iterator."""
        iterator = iter([1, 2, 3])

        adapter = _auto_adapt_data_source(iterator)

        assert isinstance(adapter, CallableAdapter)
        # Should have consumed and cached the iterator
        items = adapter.get_data_items()
        assert items == [1, 2, 3]

    def test_auto_adapt_unsupported_type(self):
        """Test auto-adapting unsupported type."""
        unsupported = 42

        with pytest.raises(ValueError, match="Unable to auto-adapt"):
            _auto_adapt_data_source(unsupported)

    def test_auto_adapt_callable_exception(self):
        """Test handling exceptions in callable auto-detection."""

        def failing_callable():
            raise Exception("Failed to call")

        with pytest.raises(ValueError, match="Unable to auto-adapt"):
            _auto_adapt_data_source(failing_callable)

    @patch("robodm.ingestion.factory.ParallelDataIngester")
    @patch("robodm.ingestion.factory.tempfile.mkdtemp")
    def test_create_vla_dataset_from_source(self, mock_mkdtemp,
                                            mock_parallel_ingester):
        """Test main factory function."""
        mock_mkdtemp.return_value = "/tmp/robodm_test"
        mock_ingester_instance = Mock()
        mock_parallel_ingester.return_value = mock_ingester_instance
        mock_ingester_instance.ingest_data.return_value = "mock_result"

        dataset = MockPyTorchDataset(5)

        result = create_vla_dataset_from_source(dataset,
                                                output_directory="/custom/dir",
                                                num_workers=8)

        assert result == "mock_result"
        mock_parallel_ingester.assert_called_once()
        config = mock_parallel_ingester.call_args[0][0]
        assert config.output_directory == "/custom/dir"
        assert config.num_workers == 8

    def test_create_vla_dataset_from_pytorch_dataset(self):
        """Test PyTorch dataset factory function."""
        dataset = MockPyTorchDataset(100)

        with patch("robodm.ingestion.factory.create_vla_dataset_from_source"
                   ) as mock_create:
            create_vla_dataset_from_pytorch_dataset(dataset,
                                                    trajectories_per_dataset=5,
                                                    num_workers=4)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["data_source"] == dataset
        assert call_kwargs["group_size"] == 20  # 100 / 5
        assert call_kwargs["num_workers"] == 4

    def test_create_vla_dataset_from_file_list(self):
        """Test file list factory function."""
        file_paths = ["file1.txt", "file2.txt"]

        def transform_fn(path):
            return {"file": path}

        with patch("robodm.ingestion.factory.create_vla_dataset_from_source"
                   ) as mock_create:
            create_vla_dataset_from_file_list(file_paths,
                                              transform_fn,
                                              files_per_trajectory=50)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["data_source"] == file_paths
        assert call_kwargs["transform_fn"] == transform_fn
        assert call_kwargs["group_size"] == 50


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestParallelDataIngester:
    """Test ParallelDataIngester class."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_setup(self):
        """Setup Ray for testing."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield
        if ray.is_initialized():
            ray.shutdown()

    def test_init_without_ray(self):
        """Test initialization when Ray is not available."""
        with patch("robodm.ingestion.parallel.RAY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Ray is required"):
                ParallelDataIngester(IngestionConfig(output_directory="/tmp"))

    @patch("robodm.ingestion.parallel.ray.is_initialized", return_value=False)
    @patch("robodm.ingestion.parallel.ray.init")
    def test_init_ray_initialization(self, mock_ray_init, mock_is_initialized,
                                     sample_config):
        """Test Ray initialization when not already initialized."""
        sample_config.ray_init_kwargs = {"local_mode": True}

        ParallelDataIngester(sample_config)

        mock_ray_init.assert_called_once_with(local_mode=True)

    @patch("robodm.ingestion.parallel.os.makedirs")
    def test_init_creates_output_directory(self, mock_makedirs, sample_config):
        """Test that output directory is created."""
        ParallelDataIngester(sample_config)

        mock_makedirs.assert_called_once_with(sample_config.output_directory,
                                              exist_ok=True)

    def test_ingest_data_empty_items(self, sample_config):
        """Test ingestion with empty data items."""
        ingester = MockDataIngester([])  # Empty items
        parallel_ingester = ParallelDataIngester(sample_config)

        with patch("robodm.ingestion.parallel.logger") as mock_logger:
            result = parallel_ingester.ingest_data(ingester,
                                                   return_vla_dataset=False)

        assert result == []
        mock_logger.warning.assert_called_with("No data items found")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_pytorch_dataset_adapter_tuple_data(self):
        """Test PyTorchDatasetAdapter with tuple data format."""

        class TupleDataset:

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return (np.array([idx]), idx)

        adapter = PyTorchDatasetAdapter(TupleDataset())
        result = adapter.transform_item(1)

        assert "input" in result
        assert "label" in result
        assert result["label"] == 1

    def test_iterator_adapter_empty_iterator(self):
        """Test IteratorAdapter with empty iterator."""

        def empty_iterator():
            return iter([])

        adapter = IteratorAdapter(empty_iterator)
        items = adapter.get_data_items()
        groups = adapter.group_items_into_trajectories(items)

        assert items == []
        assert groups == []

    def test_file_list_adapter_complex_paths(self):
        """Test FileListAdapter with complex file paths."""
        complex_paths = [
            "/very/long/path/with/many/subdirs/file.json",
            "/path/with spaces/file name.txt",
            "/path/with-dashes/file_with_underscores.data",
        ]

        def transform_fn(path):
            return {"path": path}

        adapter = FileListAdapter(complex_paths, transform_fn)
        filename = adapter.get_trajectory_filename([complex_paths[0]], 0)

        assert "file" in filename
        assert "000000" in filename

    def test_trajectory_builder_validation_failure(self, sample_config,
                                                   mock_trajectory, temp_dir):
        """Test trajectory builder with validation failures."""

        class ValidatingIngester(MockDataIngester):

            def validate_transformed_data(self, data):
                return "bad" not in data.get("data", "")

        builder = TrajectoryBuilder(sample_config)
        ingester = ValidatingIngester(
            ["good_item", "bad_item", "another_good"])
        output_path = str(temp_dir / "test.mkv")

        with patch("robodm.ingestion.base.logger") as mock_logger:
            result = builder.create_trajectory_from_group(
                ["good_item", "bad_item", "another_good"], ingester,
                output_path)

        # Should skip the 'bad_item'
        assert mock_trajectory.add_by_dict.call_count == 2
        mock_logger.debug.assert_called_once()

    def test_large_group_sizes(self):
        """Test handling of large group sizes."""
        dataset = MockPyTorchDataset(1000)
        adapter = PyTorchDatasetAdapter(dataset, group_size=500)

        items = adapter.get_data_items()
        groups = adapter.group_items_into_trajectories(items)

        assert len(groups) == 2
        assert len(groups[0]) == 500
        assert len(groups[1]) == 500

    def test_trajectory_filename_with_special_characters(self):
        """Test trajectory filename generation with special characters."""

        def transform_fn(path):
            return {"file": path}

        special_files = ["/path/file with spaces & symbols!@#.txt"]
        adapter = FileListAdapter(special_files, transform_fn)

        filename = adapter.get_trajectory_filename(special_files, 0)

        # Should handle special characters gracefully
        assert "file" in filename
        assert "000000" in filename
