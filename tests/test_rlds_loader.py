"""Tests for the RLDS loader."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from robodm.loader.rlds import RLDSLoader


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow modules."""
    with patch.dict("sys.modules", {
            "tensorflow": Mock(),
            "tensorflow_datasets": Mock()
    }):
        yield


@pytest.fixture
def mock_tfds_builder():
    """Mock TensorFlow Datasets builder."""
    mock_builder = Mock()
    mock_dataset = Mock()
    mock_builder.as_dataset.return_value = mock_dataset

    # Mock dataset length
    mock_dataset.__len__ = Mock(return_value=100)

    # Mock dataset methods
    mock_dataset.repeat.return_value = mock_dataset
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.take.return_value = mock_dataset
    mock_dataset.skip.return_value = mock_dataset

    return mock_builder


@pytest.fixture
def sample_trajectory_data():
    """Sample trajectory data structure."""
    return {
        "steps": [
            {
                "observation": {
                    "image": np.random.rand(64, 64, 3),
                    "state": np.array([0.1, 0.2, 0.3]),
                },
                "action": np.array([1.0, -1.0]),
                "reward": np.array([0.5]),
                "is_terminal": np.array([False]),
            },
            {
                "observation": {
                    "image": np.random.rand(64, 64, 3),
                    "state": np.array([0.2, 0.3, 0.4]),
                },
                "action": np.array([0.5, -0.5]),
                "reward": np.array([1.0]),
                "is_terminal": np.array([True]),
            },
        ]
    }


class TestRLDSLoader:
    """Test RLDSLoader class."""

    def test_init_without_tensorflow(self):
        """Test initialization when TensorFlow is not available."""
        with patch.dict("sys.modules", {"tensorflow": None}):
            with pytest.raises(
                    ImportError,
                    match="Please install tensorflow and tensorflow_datasets"):
                RLDSLoader("/path/to/dataset")

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_init_basic(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test basic initialization."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset",
                            split="train",
                            batch_size=4,
                            shuffling=False)

        assert loader.path == "/path/to/dataset"
        assert loader.batch_size == 4
        assert loader.split == "train"
        assert loader.length == 100
        assert loader.shuffling is False
        assert loader.index == 0

        mock_tfds.builder_from_directory.assert_called_once_with(
            "/path/to/dataset")
        mock_tfds_builder.as_dataset.assert_called_once_with("train")

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_init_with_shuffling(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test initialization with shuffling enabled."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset",
                            shuffling=True,
                            shuffle_buffer=20)

        assert loader.shuffling is True
        # Verify shuffle and repeat were called
        mock_tfds_builder.as_dataset.return_value.repeat.assert_called_once()
        mock_tfds_builder.as_dataset.return_value.shuffle.assert_called_once_with(
            20)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_len(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test __len__ method."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset")

        assert len(loader) == 100

    def test_len_without_tensorflow(self):
        """Test __len__ when TensorFlow is not available."""
        # Create a mock loader without proper TensorFlow setup
        loader = object.__new__(RLDSLoader)
        loader.length = 50

        with patch.dict("sys.modules", {"tensorflow": None}):
            with pytest.raises(ImportError, match="Please install tensorflow"):
                len(loader)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_iter(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test __iter__ method."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset")

        assert iter(loader) is loader

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_get_batch(self, mock_tf, mock_tfds, mock_tfds_builder,
                       sample_trajectory_data):
        """Test get_batch method."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        # Mock the batch data
        mock_batch = [sample_trajectory_data, sample_trajectory_data]
        mock_tfds_builder.as_dataset.return_value.take.return_value = mock_batch

        loader = RLDSLoader("/path/to/dataset", batch_size=2, shuffling=False)

        with patch.object(
                loader,
                "_convert_traj_to_numpy",
                side_effect=lambda x: f"converted_{id(x)}") as mock_convert:
            batch = loader.get_batch()

        assert len(batch) == 2
        assert loader.index == 2
        assert mock_convert.call_count == 2

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_get_batch_stop_iteration(self, mock_tf, mock_tfds,
                                      mock_tfds_builder):
        """Test get_batch raises StopIteration when no shuffling and at end."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset", batch_size=10, shuffling=False)
        loader.index = 95  # Near the end

        mock_batch = [{}] * 10
        mock_tfds_builder.as_dataset.return_value.take.return_value = mock_batch

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value="converted"):
            batch = loader.get_batch()
            # After this batch, index will be 105 > length (100)
            assert loader.index == 105

            # Next call should raise StopIteration
            with pytest.raises(StopIteration):
                loader.get_batch()

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_next(self, mock_tf, mock_tfds, mock_tfds_builder,
                  sample_trajectory_data):
        """Test __next__ method."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        # Mock the iterator
        mock_iterator = Mock()
        mock_iterator.__next__ = Mock(return_value=sample_trajectory_data)

        loader = RLDSLoader("/path/to/dataset", shuffling=False)
        loader.iterator = mock_iterator

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value="converted_traj") as mock_convert:
            result = next(loader)

        assert result == ["converted_traj"]
        assert loader.index == 1
        mock_convert.assert_called_once_with(sample_trajectory_data)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_next_stop_iteration(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test __next__ raises StopIteration at end."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset", shuffling=False)
        loader.index = 99  # At the end

        mock_iterator = Mock()
        mock_iterator.__next__ = Mock(return_value={})
        loader.iterator = mock_iterator

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value="converted"):
            result = next(loader)
            assert loader.index == 100

            # Next call should raise StopIteration
            with pytest.raises(StopIteration):
                next(loader)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_getitem(self, mock_tf, mock_tfds, mock_tfds_builder,
                     sample_trajectory_data):
        """Test __getitem__ method."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        # Mock the dataset skip/take operations
        mock_dataset = mock_tfds_builder.as_dataset.return_value
        mock_skip_take = Mock()
        mock_skip_take.__iter__ = Mock(
            return_value=iter([sample_trajectory_data]))
        mock_dataset.skip.return_value.take.return_value = mock_skip_take

        loader = RLDSLoader("/path/to/dataset")

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value="converted_item") as mock_convert:
            result = loader[5]

        assert result == "converted_item"
        mock_dataset.skip.assert_called_once_with(5)
        mock_dataset.skip.return_value.take.assert_called_once_with(1)
        mock_convert.assert_called_once_with(sample_trajectory_data)

    def test_convert_traj_to_numpy_simple(self, sample_trajectory_data):
        """Test _convert_traj_to_numpy with simple data."""
        loader = object.__new__(RLDSLoader)  # Create without __init__

        with patch("robodm.loader.rlds.tf"):
            result = loader._convert_traj_to_numpy(sample_trajectory_data)

        assert isinstance(result, list)
        assert len(result) == 2  # Two steps

        # Check first step
        step1 = result[0]
        assert "observation" in step1
        assert "action" in step1
        assert "reward" in step1
        assert "is_terminal" in step1

        # Check that observation is a dict with numpy arrays
        assert isinstance(step1["observation"], dict)
        assert "image" in step1["observation"]
        assert "state" in step1["observation"]
        assert isinstance(step1["observation"]["image"], np.ndarray)
        assert isinstance(step1["observation"]["state"], np.ndarray)

        # Check other fields are numpy arrays
        assert isinstance(step1["action"], np.ndarray)
        assert isinstance(step1["reward"], np.ndarray)
        assert isinstance(step1["is_terminal"], np.ndarray)

    def test_convert_traj_to_numpy_flat_structure(self):
        """Test _convert_traj_to_numpy with flat structure."""
        flat_traj = {
            "steps": [{
                "action": np.array([1.0, 2.0]),
                "reward": np.array([0.5])
            }]
        }

        loader = object.__new__(RLDSLoader)

        with patch("robodm.loader.rlds.tf"):
            result = loader._convert_traj_to_numpy(flat_traj)

        assert len(result) == 1
        step = result[0]
        assert "action" in step
        assert "reward" in step
        assert isinstance(step["action"], np.ndarray)
        assert isinstance(step["reward"], np.ndarray)

    def test_convert_traj_to_numpy_nested_dict(self):
        """Test _convert_traj_to_numpy with deeply nested dictionaries."""
        nested_traj = {
            "steps": [{
                "observation": {
                    "sensors": {
                        "camera": np.array([1, 2, 3]),
                        "lidar": np.array([4, 5, 6]),
                    },
                    "proprioception": {
                        "joint_pos": np.array([0.1, 0.2]),
                        "joint_vel": np.array([1.0, 2.0]),
                    },
                },
                "action": np.array([0.5]),
            }]
        }

        loader = object.__new__(RLDSLoader)

        with patch("robodm.loader.rlds.tf"):
            result = loader._convert_traj_to_numpy(nested_traj)

        step = result[0]

        # Check nested structure is preserved
        assert "observation" in step
        obs = step["observation"]
        assert "sensors" in obs
        assert "proprioception" in obs

        # Check sensors
        sensors = obs["sensors"]
        assert "camera" in sensors
        assert "lidar" in sensors
        assert isinstance(sensors["camera"], np.ndarray)
        assert isinstance(sensors["lidar"], np.ndarray)

        # Check proprioception
        proprio = obs["proprioception"]
        assert "joint_pos" in proprio
        assert "joint_vel" in proprio
        assert isinstance(proprio["joint_pos"], np.ndarray)
        assert isinstance(proprio["joint_vel"], np.ndarray)


class TestRLDSLoaderEdgeCases:
    """Test edge cases for RLDS loader."""

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_empty_trajectory(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test handling of empty trajectory."""
        empty_traj = {"steps": []}

        mock_tfds.builder_from_directory.return_value = mock_tfds_builder
        loader = RLDSLoader("/path/to/dataset")

        with patch("robodm.loader.rlds.tf"):
            result = loader._convert_traj_to_numpy(empty_traj)

        assert result == []

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_zero_batch_size(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test with zero batch size."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset", batch_size=0)

        assert loader.batch_size == 0

        # Mock empty batch
        mock_tfds_builder.as_dataset.return_value.take.return_value = []

        batch = loader.get_batch()
        assert batch == []

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_different_splits(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test with different dataset splits."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        # Test different splits
        for split in ["train", "test", "validation"]:
            loader = RLDSLoader("/path/to/dataset", split=split)
            assert loader.split == split
            mock_tfds_builder.as_dataset.assert_called_with(split)

    def test_convert_traj_to_numpy_mixed_types(self):
        """Test _convert_traj_to_numpy with mixed data types."""
        mixed_traj = {
            "steps": [{
                "string_field": "text_data",
                "int_field": 42,
                "float_field": 3.14,
                "array_field": np.array([1, 2, 3]),
                "nested": {
                    "inner_string": "inner_text",
                    "inner_array": np.array([4, 5, 6]),
                },
            }]
        }

        loader = object.__new__(RLDSLoader)

        with patch("robodm.loader.rlds.tf"):
            result = loader._convert_traj_to_numpy(mixed_traj)

        step = result[0]

        # All fields should be converted to numpy arrays or dict of numpy arrays
        assert isinstance(step["string_field"], np.ndarray)
        assert isinstance(step["int_field"], np.ndarray)
        assert isinstance(step["float_field"], np.ndarray)
        assert isinstance(step["array_field"], np.ndarray)

        # Nested dict should preserve structure
        assert isinstance(step["nested"], dict)
        assert isinstance(step["nested"]["inner_string"], np.ndarray)
        assert isinstance(step["nested"]["inner_array"], np.ndarray)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_large_shuffle_buffer(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test with large shuffle buffer."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset",
                            shuffle_buffer=10000,
                            shuffling=True)

        # Verify shuffle was called with large buffer
        mock_tfds_builder.as_dataset.return_value.shuffle.assert_called_once_with(
            10000)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_index_tracking_with_shuffling(self, mock_tf, mock_tfds,
                                           mock_tfds_builder):
        """Test index tracking with shuffling enabled."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset", shuffling=True)

        # With shuffling, should not raise StopIteration based on index
        loader.index = 150  # Beyond original length

        mock_iterator = Mock()
        mock_iterator.__next__ = Mock(return_value={"steps": []})
        loader.iterator = mock_iterator

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value="converted"):
            # Should not raise StopIteration because shuffling=True
            result = next(loader)
            assert result == ["converted"]


class TestRLDSLoaderIntegration:
    """Test integration scenarios for RLDS loader."""

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_full_iteration_cycle(self, mock_tf, mock_tfds, mock_tfds_builder):
        """Test full iteration cycle without shuffling."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        # Create loader with small dataset
        mock_tfds_builder.as_dataset.return_value.__len__ = Mock(
            return_value=3)
        loader = RLDSLoader("/path/to/dataset", shuffling=False)
        loader.length = 3

        # Mock iterator
        sample_data = {"steps": [{"action": np.array([1.0])}]}
        mock_iterator = Mock()
        mock_iterator.__next__ = Mock(
            side_effect=[sample_data, sample_data, sample_data, StopIteration])
        loader.iterator = mock_iterator

        with patch.object(loader,
                          "_convert_traj_to_numpy",
                          return_value=["converted"]):
            # Should be able to iterate through all items
            items = []
            try:
                while True:
                    items.append(next(loader))
            except StopIteration:
                pass

            assert len(items) == 3
            assert all(item == ["converted"] for item in items)

    @patch("robodm.loader.rlds.tfds")
    @patch("robodm.loader.rlds.tf")
    def test_batch_and_single_item_consistency(self, mock_tf, mock_tfds,
                                               mock_tfds_builder,
                                               sample_trajectory_data):
        """Test that batch and single item access return consistent data."""
        mock_tfds.builder_from_directory.return_value = mock_tfds_builder

        loader = RLDSLoader("/path/to/dataset", batch_size=1)

        # Mock single item access
        mock_dataset = mock_tfds_builder.as_dataset.return_value
        mock_skip_take = Mock()
        mock_skip_take.__iter__ = Mock(
            return_value=iter([sample_trajectory_data]))
        mock_dataset.skip.return_value.take.return_value = mock_skip_take

        # Mock batch access
        mock_dataset.take.return_value = [sample_trajectory_data]

        with patch.object(
                loader,
                "_convert_traj_to_numpy",
                side_effect=lambda x: f"converted_{id(x)}") as mock_convert:
            # Get single item
            single_item = loader[0]

            # Get batch
            batch = loader.get_batch()

        # Both should have called convert function
        assert mock_convert.call_count == 2

        # Batch should contain one item (since batch_size=1)
        assert len(batch) == 1
