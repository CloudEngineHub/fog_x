"""Tests for the VLADataset system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

try:
    import ray
    import ray.data as rd

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from robodm.dataset import (DatasetConfig, VLADataset, load_slice_dataset,
                            load_trajectory_dataset, split_dataset)
from robodm.loader.vla import LoadingMode, SliceConfig


@pytest.fixture(scope="session", autouse=True)
def ray_setup():
    """Setup Ray for testing if available."""
    if RAY_AVAILABLE and not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def mock_ray_vla_loader():
    """Mock RayVLALoader for testing."""
    with patch("robodm.dataset.RayVLALoader") as mock_loader_class:
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock dataset methods
        mock_dataset = Mock()
        mock_loader.dataset = mock_dataset
        mock_loader.count.return_value = 100
        mock_loader.peek.return_value = {
            "observation/images/cam_high": np.random.rand(10, 128, 128, 3),
            "action": np.random.rand(10, 7),
        }
        mock_loader.schema.return_value = {
            "observation/images/cam_high": {
                "shape": (10, 128, 128, 3),
                "dtype": "float32",
            },
            "action": {
                "shape": (10, 7),
                "dtype": "float32"
            },
        }
        mock_loader.take.return_value = [mock_loader.peek()]
        mock_loader.sample.return_value = [mock_loader.peek()]
        mock_loader.iter_batches.return_value = iter([mock_loader.peek()])
        mock_loader.iter_rows.return_value = iter([mock_loader.peek()])
        mock_loader.materialize.return_value = [mock_loader.peek()]
        mock_loader.split.return_value = [mock_dataset, mock_dataset]

        yield mock_loader_class


@pytest.fixture
def sample_vla_files(temp_dir):
    """Create sample VLA files for testing."""
    # Create some dummy VLA files
    vla_files = []
    for i in range(3):
        vla_path = temp_dir / f"trajectory_{i}.vla"
        vla_path.touch()
        vla_files.append(str(vla_path))
    return vla_files


class TestDatasetConfig:
    """Test DatasetConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.batch_size == 1
        assert config.shuffle is False
        assert config.num_parallel_reads == 4
        assert config.ray_init_kwargs is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DatasetConfig(
            batch_size=32,
            shuffle=True,
            num_parallel_reads=8,
            ray_init_kwargs={"local_mode": True},
        )
        assert config.batch_size == 32
        assert config.shuffle is True
        assert config.num_parallel_reads == 8
        assert config.ray_init_kwargs == {"local_mode": True}


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestVLADataset:
    """Test VLADataset class."""

    def test_init_without_ray_available(self):
        """Test initialization when Ray is not available."""
        with patch("robodm.dataset.RAY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Ray is required"):
                VLADataset("/path/to/data")

    def test_init_trajectory_mode(self, mock_ray_vla_loader, sample_vla_files):
        """Test initialization in trajectory mode."""
        dataset = VLADataset(path=sample_vla_files[0],
                             mode="trajectory",
                             return_type="numpy")

        assert dataset.path == sample_vla_files[0]
        assert dataset.mode == LoadingMode.TRAJECTORY
        assert dataset.return_type == "numpy"
        assert isinstance(dataset.config, DatasetConfig)
        assert dataset._schema is None
        assert dataset._stats is None

        # Verify loader was called with correct parameters
        mock_ray_vla_loader.assert_called_once()
        call_args = mock_ray_vla_loader.call_args
        assert call_args[1]["path"] == sample_vla_files[0]
        assert call_args[1]["mode"] == LoadingMode.TRAJECTORY
        assert call_args[1]["return_type"] == "numpy"

    def test_init_slice_mode(self, mock_ray_vla_loader, sample_vla_files):
        """Test initialization in slice mode."""
        slice_config = SliceConfig(slice_length=50)
        dataset = VLADataset(path=sample_vla_files[0],
                             mode=LoadingMode.SLICE,
                             slice_config=slice_config)

        assert dataset.mode == LoadingMode.SLICE
        mock_ray_vla_loader.assert_called_once()
        call_args = mock_ray_vla_loader.call_args
        assert call_args[1]["slice_config"] == slice_config

    def test_init_custom_config(self, mock_ray_vla_loader, sample_vla_files):
        """Test initialization with custom config."""
        config = DatasetConfig(batch_size=16, shuffle=True)
        dataset = VLADataset(path=sample_vla_files[0], config=config)

        assert dataset.config == config
        mock_ray_vla_loader.assert_called_once()
        call_args = mock_ray_vla_loader.call_args
        assert call_args[1]["batch_size"] == 16
        assert call_args[1]["shuffle"] is True

    @patch("robodm.dataset.ray.is_initialized", return_value=False)
    @patch("robodm.dataset.ray.init")
    def test_ray_initialization(self, mock_ray_init, mock_is_initialized,
                                mock_ray_vla_loader, sample_vla_files):
        """Test Ray initialization when not already initialized."""
        config = DatasetConfig(ray_init_kwargs={"local_mode": True})
        VLADataset(path=sample_vla_files[0], config=config)

        mock_ray_init.assert_called_once_with(local_mode=True)

    def test_create_trajectory_dataset(self, mock_ray_vla_loader,
                                       sample_vla_files):
        """Test create_trajectory_dataset class method."""
        dataset = VLADataset.create_trajectory_dataset(
            path=sample_vla_files[0], return_type="tensor")

        assert dataset.mode == LoadingMode.TRAJECTORY
        assert dataset.return_type == "tensor"
        mock_ray_vla_loader.assert_called_once()

    def test_create_slice_dataset(self, mock_ray_vla_loader, sample_vla_files):
        """Test create_slice_dataset class method."""
        dataset = VLADataset.create_slice_dataset(path=sample_vla_files[0],
                                                  slice_length=100,
                                                  stride=2,
                                                  random_start=False)

        assert dataset.mode == LoadingMode.SLICE
        mock_ray_vla_loader.assert_called_once()
        call_args = mock_ray_vla_loader.call_args
        slice_config = call_args[1]["slice_config"]
        assert slice_config.slice_length == 100
        assert slice_config.stride == 2
        assert slice_config.random_start is False

    def test_get_ray_dataset(self, mock_ray_vla_loader, sample_vla_files):
        """Test get_ray_dataset method."""
        dataset = VLADataset(path=sample_vla_files[0])
        ray_dataset = dataset.get_ray_dataset()

        assert ray_dataset == dataset.loader.dataset

    def test_iter_batches(self, mock_ray_vla_loader, sample_vla_files):
        """Test iter_batches method."""
        dataset = VLADataset(path=sample_vla_files[0])
        batches = list(dataset.iter_batches())

        dataset.loader.iter_batches.assert_called_once_with(None)
        assert len(batches) == 1

    def test_iter_rows(self, mock_ray_vla_loader, sample_vla_files):
        """Test iter_rows method."""
        dataset = VLADataset(path=sample_vla_files[0])
        rows = list(dataset.iter_rows())

        dataset.loader.iter_rows.assert_called_once()
        assert len(rows) == 1

    def test_take(self, mock_ray_vla_loader, sample_vla_files):
        """Test take method."""
        dataset = VLADataset(path=sample_vla_files[0])
        items = dataset.take(5)

        dataset.loader.take.assert_called_once_with(5)
        assert len(items) == 1

    def test_sample(self, mock_ray_vla_loader, sample_vla_files):
        """Test sample method."""
        dataset = VLADataset(path=sample_vla_files[0])
        samples = dataset.sample(3, replace=True)

        dataset.loader.sample.assert_called_once_with(3, True)
        assert len(samples) == 1

    def test_count(self, mock_ray_vla_loader, sample_vla_files):
        """Test count method."""
        dataset = VLADataset(path=sample_vla_files[0])
        count = dataset.count()

        dataset.loader.count.assert_called_once()
        assert count == 100

    def test_schema(self, mock_ray_vla_loader, sample_vla_files):
        """Test schema method with caching."""
        dataset = VLADataset(path=sample_vla_files[0])

        # First call should fetch schema
        schema1 = dataset.schema()
        dataset.loader.schema.assert_called_once()

        # Second call should use cached schema
        schema2 = dataset.schema()
        dataset.loader.schema.assert_called_once()  # Still only called once

        assert schema1 == schema2
        assert dataset._schema is not None

    def test_split(self, mock_ray_vla_loader, sample_vla_files):
        """Test split method."""
        dataset = VLADataset(path=sample_vla_files[0])
        splits = dataset.split(0.7, 0.3, shuffle=True)

        dataset.loader.split.assert_called_once_with(0.7, 0.3, shuffle=True)
        assert len(splits) == 2
        assert all(isinstance(split, VLADataset) for split in splits)

        # Verify split datasets have correct properties
        for split in splits:
            assert split.path == dataset.path
            assert split.mode == dataset.mode
            assert split.return_type == dataset.return_type
            assert split.config == dataset.config

    def test_filter(self, mock_ray_vla_loader, sample_vla_files):
        """Test filter method."""
        dataset = VLADataset(path=sample_vla_files[0])
        filter_fn = lambda x: len(x["action"]) > 5
        filtered = dataset.filter(filter_fn)

        dataset.loader.dataset.filter.assert_called_once_with(filter_fn)
        assert isinstance(filtered, VLADataset)
        assert filtered.path == dataset.path
        assert filtered._schema == dataset._schema

    def test_map(self, mock_ray_vla_loader, sample_vla_files):
        """Test map method."""
        dataset = VLADataset(path=sample_vla_files[0])
        map_fn = lambda x: {"action": x["action"] * 2}
        mapped = dataset.map(map_fn, batch_format="numpy")

        dataset.loader.dataset.map.assert_called_once_with(
            map_fn, batch_format="numpy")
        assert isinstance(mapped, VLADataset)
        assert mapped.path == dataset.path
        assert mapped._schema is None  # Schema should be reset

    def test_shuffle(self, mock_ray_vla_loader, sample_vla_files):
        """Test shuffle method."""
        dataset = VLADataset(path=sample_vla_files[0])
        shuffled = dataset.shuffle(seed=42)

        dataset.loader.dataset.random_shuffle.assert_called_once_with(seed=42)
        assert isinstance(shuffled, VLADataset)
        assert shuffled.path == dataset.path

    def test_materialize(self, mock_ray_vla_loader, sample_vla_files):
        """Test materialize method."""
        dataset = VLADataset(path=sample_vla_files[0])
        materialized = dataset.materialize()

        dataset.loader.materialize.assert_called_once()
        assert len(materialized) == 1

    def test_get_stats_trajectory_mode(self, mock_ray_vla_loader,
                                       sample_vla_files):
        """Test get_stats for trajectory mode."""
        dataset = VLADataset(path=sample_vla_files[0],
                             mode=LoadingMode.TRAJECTORY)
        stats = dataset.get_stats()

        expected_keys = [
            "mode",
            "return_type",
            "total_items",
            "sample_keys",
            "trajectory_length",
        ]
        assert all(key in stats for key in expected_keys)
        assert stats["mode"] == "trajectory"
        assert stats["total_items"] == 100
        assert stats["trajectory_length"] == 10
        assert dataset._stats is not None

    def test_get_stats_slice_mode(self, mock_ray_vla_loader, sample_vla_files):
        """Test get_stats for slice mode."""
        dataset = VLADataset(path=sample_vla_files[0], mode=LoadingMode.SLICE)
        stats = dataset.get_stats()

        expected_keys = [
            "mode",
            "return_type",
            "total_items",
            "sample_keys",
            "slice_length",
        ]
        assert all(key in stats for key in expected_keys)
        assert stats["mode"] == "slice"
        assert stats["slice_length"] == 10

    def test_get_stats_empty_dataset(self, mock_ray_vla_loader,
                                     sample_vla_files):
        """Test get_stats for empty dataset."""
        dataset = VLADataset(path=sample_vla_files[0])
        dataset.loader.peek.return_value = None
        stats = dataset.get_stats()

        assert stats == {"mode": "trajectory", "total_items": 0}

    def test_peek(self, mock_ray_vla_loader, sample_vla_files):
        """Test peek method."""
        dataset = VLADataset(path=sample_vla_files[0])
        sample = dataset.peek()

        dataset.loader.peek.assert_called_once()
        assert "observation/images/cam_high" in sample
        assert "action" in sample

    def test_get_tf_schema(self, mock_ray_vla_loader, sample_vla_files):
        """Test get_tf_schema method."""
        with patch("robodm.dataset.data_to_tf_schema") as mock_schema_fn:
            mock_schema_fn.return_value = {"action": "tf.float32"}

            dataset = VLADataset(path=sample_vla_files[0])
            schema = dataset.get_tf_schema()

            mock_schema_fn.assert_called_once()
            assert schema == {"action": "tf.float32"}

    def test_get_tf_schema_empty(self, mock_ray_vla_loader, sample_vla_files):
        """Test get_tf_schema with empty dataset."""
        dataset = VLADataset(path=sample_vla_files[0])
        dataset.loader.peek.return_value = None
        schema = dataset.get_tf_schema()

        assert schema is None

    def test_iterator_protocol(self, mock_ray_vla_loader, sample_vla_files):
        """Test iterator protocol."""
        dataset = VLADataset(path=sample_vla_files[0])
        items = list(dataset)

        assert len(items) == 1

    def test_len(self, mock_ray_vla_loader, sample_vla_files):
        """Test __len__ method."""
        dataset = VLADataset(path=sample_vla_files[0])
        assert len(dataset) == 100

    def test_getitem_not_supported(self, mock_ray_vla_loader,
                                   sample_vla_files):
        """Test that __getitem__ raises NotImplementedError."""
        dataset = VLADataset(path=sample_vla_files[0])
        with pytest.raises(NotImplementedError,
                           match="Random access not supported"):
            _ = dataset[0]

    def test_legacy_methods(self, mock_ray_vla_loader, sample_vla_files):
        """Test legacy compatibility methods."""
        dataset = VLADataset(path=sample_vla_files[0])

        # Test get_loader
        loader = dataset.get_loader()
        assert loader == dataset.loader

        # Test get_next_trajectory
        with patch.object(dataset, "__next__") as mock_next:
            mock_next.return_value = {"action": np.array([1, 2, 3])}
            traj = dataset.get_next_trajectory()
            assert "action" in traj


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_load_trajectory_dataset(self, mock_ray_vla_loader,
                                     sample_vla_files):
        """Test load_trajectory_dataset function."""
        dataset = load_trajectory_dataset(path=sample_vla_files[0],
                                          batch_size=16,
                                          shuffle=True,
                                          return_type="tensor")

        assert isinstance(dataset, VLADataset)
        assert dataset.mode == LoadingMode.TRAJECTORY
        assert dataset.return_type == "tensor"
        assert dataset.config.batch_size == 16
        assert dataset.config.shuffle is True

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_load_slice_dataset(self, mock_ray_vla_loader, sample_vla_files):
        """Test load_slice_dataset function."""
        dataset = load_slice_dataset(path=sample_vla_files[0],
                                     slice_length=200,
                                     stride=5,
                                     batch_size=8)

        assert isinstance(dataset, VLADataset)
        assert dataset.mode == LoadingMode.SLICE
        assert dataset.config.batch_size == 8

        # Verify slice config was passed correctly
        mock_ray_vla_loader.assert_called_once()
        call_args = mock_ray_vla_loader.call_args
        slice_config = call_args[1]["slice_config"]
        assert slice_config.slice_length == 200
        assert slice_config.stride == 5

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_split_dataset(self, mock_ray_vla_loader, sample_vla_files):
        """Test split_dataset function."""
        dataset = VLADataset(path=sample_vla_files[0])
        train_ds, val_ds = split_dataset(dataset, 0.8, 0.2, shuffle=True)

        assert isinstance(train_ds, VLADataset)
        assert isinstance(val_ds, VLADataset)
        dataset.loader.split.assert_called_once_with(0.8, 0.2, shuffle=True)

    def test_split_dataset_invalid_fractions(self, mock_ray_vla_loader,
                                             sample_vla_files):
        """Test split_dataset with invalid fractions."""
        dataset = VLADataset(path=sample_vla_files[0])

        with pytest.raises(ValueError, match="must equal 1.0"):
            split_dataset(dataset, 0.6, 0.3)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_string_mode_conversion(self, mock_ray_vla_loader,
                                    sample_vla_files):
        """Test conversion of string mode to LoadingMode enum."""
        # Test trajectory mode
        dataset1 = VLADataset(path=sample_vla_files[0], mode="trajectory")
        assert dataset1.mode == LoadingMode.TRAJECTORY

        # Test slice mode
        dataset2 = VLADataset(path=sample_vla_files[0], mode="slice")
        assert dataset2.mode == LoadingMode.SLICE

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_empty_path_handling(self, mock_ray_vla_loader):
        """Test handling of empty or invalid paths."""
        # Should not raise error during initialization
        dataset = VLADataset(path="")
        assert dataset.path == ""

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_multiple_operations_chaining(self, mock_ray_vla_loader,
                                          sample_vla_files):
        """Test chaining multiple dataset operations."""
        dataset = VLADataset(path=sample_vla_files[0])

        # Chain multiple operations
        processed = dataset.filter(lambda x: True).map(lambda x: x).shuffle(
            seed=42)

        assert isinstance(processed, VLADataset)
        assert processed.path == dataset.path

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
    def test_stats_caching(self, mock_ray_vla_loader, sample_vla_files):
        """Test that stats are properly cached."""
        dataset = VLADataset(path=sample_vla_files[0])

        # First call should compute stats
        stats1 = dataset.get_stats()
        dataset.loader.peek.assert_called_once()

        # Second call should use cached stats
        stats2 = dataset.get_stats()
        dataset.loader.peek.assert_called_once()  # Still only called once

        assert stats1 == stats2
        assert dataset._stats is not None
