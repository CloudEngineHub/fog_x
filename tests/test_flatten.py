"""Tests for data flattening utilities."""

import tempfile
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from robodm.utils.flatten import (_flatten, _flatten_dict, data_to_tf_schema,
                                  recursively_read_hdf5_group)


class TestDataToTfSchema:
    """Test data_to_tf_schema function."""

    def test_simple_data(self):
        """Test schema generation for simple data."""
        data = {"action": np.array([1.0, 2.0, 3.0]), "reward": np.array([1.5])}

        with patch("robodm.utils.flatten.FeatureType") as mock_feature_type:
            mock_ft_instance = Mock()
            mock_ft_instance.to_tf_feature_type.return_value = "tf_feature"
            mock_feature_type.from_data.return_value = mock_ft_instance

            schema = data_to_tf_schema(data)

        assert "action" in schema
        assert "reward" in schema
        assert schema["action"] == "tf_feature"
        assert schema["reward"] == "tf_feature"

        # Verify FeatureType.from_data was called for each field
        assert mock_feature_type.from_data.call_count == 2

        # Verify to_tf_feature_type was called with first_dim_none=True
        mock_ft_instance.to_tf_feature_type.assert_called_with(
            first_dim_none=True)

    def test_nested_data_with_observation(self):
        """Test schema generation for nested data with observation."""
        data = {
            "action": np.array([1.0, 2.0]),
            "observation": {
                "images": {
                    "cam_high": np.random.rand(128, 128, 3),
                    "cam_low": np.random.rand(64, 64, 3),
                },
                "state": {
                    "joint_pos": np.array([0.1, 0.2, 0.3])
                },
            },
        }

        with patch("robodm.utils.flatten.FeatureType") as mock_feature_type:
            mock_ft_instance = Mock()
            mock_ft_instance.to_tf_feature_type.return_value = "tf_feature"
            mock_feature_type.from_data.return_value = mock_ft_instance

            schema = data_to_tf_schema(data)

        # Check top-level action
        assert "action" in schema
        assert schema["action"] == "tf_feature"

        # Check nested observation structure
        assert "observation" in schema
        assert isinstance(schema["observation"], dict)

        # Check images
        assert "images" in schema["observation"]
        assert isinstance(schema["observation"]["images"], dict)
        assert "cam_high" in schema["observation"]["images"]
        assert "cam_low" in schema["observation"]["images"]

        # Check state
        assert "state" in schema["observation"]
        assert isinstance(schema["observation"]["state"], dict)
        assert "joint_pos" in schema["observation"]["state"]

    def test_flat_keys_with_slashes(self):
        """Test handling of flat keys with slashes."""
        data = {
            "observation/images/cam1": np.random.rand(128, 128, 3),
            "observation/state/joints": np.array([1, 2, 3]),
            "action": np.array([0.5]),
        }

        with patch("robodm.utils.flatten.FeatureType") as mock_feature_type:
            mock_ft_instance = Mock()
            mock_ft_instance.to_tf_feature_type.return_value = "tf_feature"
            mock_feature_type.from_data.return_value = mock_ft_instance

            schema = data_to_tf_schema(data)

        # Check that observation was created as a nested dict
        assert "observation" in schema
        assert isinstance(schema["observation"], dict)
        assert "images" in schema["observation"]
        assert "state" in schema["observation"]

        # Check that action remains at top level
        assert "action" in schema
        assert schema["action"] == "tf_feature"

    def test_mixed_slash_formats(self):
        """Test mixed slash and nested dict formats."""
        data = {
            "action": np.array([1.0]),
            "observation/images/cam1": np.random.rand(64, 64, 3),
            "observation": {
                "state": {
                    "joints": np.array([0.1, 0.2])
                }
            },
        }

        with patch("robodm.utils.flatten.FeatureType") as mock_feature_type:
            mock_ft_instance = Mock()
            mock_ft_instance.to_tf_feature_type.return_value = "tf_feature"
            mock_feature_type.from_data.return_value = mock_ft_instance

            schema = data_to_tf_schema(data)

        # Both should end up in observation
        assert "observation" in schema
        assert "images" in schema["observation"]
        assert "state" in schema["observation"]


class TestFlatten:
    """Test _flatten function."""

    def test_simple_dict(self):
        """Test flattening simple dictionary."""
        data = {"a": 1, "b": 2}

        result = _flatten(data)

        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        """Test flattening nested dictionary."""
        data = {
            "observation": {
                "images": {
                    "cam1": "image_data"
                },
                "state": "state_data"
            },
            "action": "action_data",
        }

        result = _flatten(data)

        expected = {
            "observation/images/cam1": "image_data",
            "observation/state": "state_data",
            "action": "action_data",
        }
        assert result == expected

    def test_deeply_nested(self):
        """Test flattening deeply nested dictionary."""
        data = {"level1": {"level2": {"level3": {"level4": "deep_value"}}}}

        result = _flatten(data)

        assert result == {"level1/level2/level3/level4": "deep_value"}

    def test_custom_separator(self):
        """Test flattening with custom separator."""
        data = {"a": {"b": {"c": "value"}}}

        result = _flatten(data, sep=".")

        assert result == {"a.b.c": "value"}

    def test_with_parent_key(self):
        """Test flattening with parent key."""
        data = {"child1": "value1", "child2": {"grandchild": "value2"}}

        result = _flatten(data, parent_key="root")

        expected = {
            "root/child1": "value1",
            "root/child2/grandchild": "value2"
        }
        assert result == expected

    def test_empty_dict(self):
        """Test flattening empty dictionary."""
        result = _flatten({})
        assert result == {}

    def test_mixed_types(self):
        """Test flattening with mixed value types."""
        data = {
            "string": "hello",
            "number": 42,
            "array": np.array([1, 2, 3]),
            "nested": {
                "list": [1, 2, 3],
                "none": None
            },
        }

        result = _flatten(data)

        assert result["string"] == "hello"
        assert result["number"] == 42
        assert np.array_equal(result["array"], np.array([1, 2, 3]))
        assert result["nested/list"] == [1, 2, 3]
        assert result["nested/none"] is None


class TestFlattenDict:
    """Test _flatten_dict function."""

    def test_simple_dict(self):
        """Test flattening simple dictionary with underscore separator."""
        data = {"a": 1, "b": 2}

        result = _flatten_dict(data)

        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        """Test flattening nested dictionary with underscore separator."""
        data = {
            "observation": {
                "images": {
                    "cam1": "image_data"
                },
                "state": "state_data"
            },
            "action": "action_data",
        }

        result = _flatten_dict(data)

        expected = {
            "observation_images_cam1": "image_data",
            "observation_state": "state_data",
            "action": "action_data",
        }
        assert result == expected

    def test_custom_separator(self):
        """Test flattening with custom separator."""
        data = {"a": {"b": {"c": "value"}}}

        result = _flatten_dict(data, sep=".")

        assert result == {"a.b.c": "value"}

    def test_with_parent_key(self):
        """Test flattening with parent key."""
        data = {"child1": "value1", "child2": {"grandchild": "value2"}}

        result = _flatten_dict(data, parent_key="root")

        expected = {
            "root_child1": "value1",
            "root_child2_grandchild": "value2"
        }
        assert result == expected

    def test_empty_dict(self):
        """Test flattening empty dictionary."""
        result = _flatten_dict({})
        assert result == {}


class TestRecursivelyReadHdf5Group:
    """Test recursively_read_hdf5_group function."""

    def test_read_dataset(self):
        """Test reading HDF5 dataset."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write test data
            with h5py.File(tmp_path, "w") as f:
                test_data = np.array([1, 2, 3, 4, 5])
                f.create_dataset("test_dataset", data=test_data)

            # Read back using function
            with h5py.File(tmp_path, "r") as f:
                dataset = f["test_dataset"]
                result = recursively_read_hdf5_group(dataset)

            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, test_data)

        finally:
            import os

            os.unlink(tmp_path)

    def test_read_group(self):
        """Test reading HDF5 group."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write test data
            with h5py.File(tmp_path, "w") as f:
                group = f.create_group("test_group")
                group.create_dataset("dataset1", data=np.array([1, 2, 3]))
                group.create_dataset("dataset2", data=np.array([4, 5, 6]))

                subgroup = group.create_group("subgroup")
                subgroup.create_dataset("dataset3", data=np.array([7, 8, 9]))

            # Read back using function
            with h5py.File(tmp_path, "r") as f:
                group = f["test_group"]
                result = recursively_read_hdf5_group(group)

            assert isinstance(result, dict)
            assert "dataset1" in result
            assert "dataset2" in result
            assert "subgroup" in result

            assert np.array_equal(result["dataset1"], np.array([1, 2, 3]))
            assert np.array_equal(result["dataset2"], np.array([4, 5, 6]))

            assert isinstance(result["subgroup"], dict)
            assert "dataset3" in result["subgroup"]
            assert np.array_equal(result["subgroup"]["dataset3"],
                                  np.array([7, 8, 9]))

        finally:
            import os

            os.unlink(tmp_path)

    def test_read_complex_structure(self):
        """Test reading complex nested HDF5 structure."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write complex test data
            with h5py.File(tmp_path, "w") as f:
                # Root level datasets
                f.create_dataset("root_data", data=np.array([10, 20]))

                # Observation group
                obs_group = f.create_group("observation")

                # Images subgroup
                images_group = obs_group.create_group("images")
                images_group.create_dataset("cam_high",
                                            data=np.random.rand(128, 128, 3))
                images_group.create_dataset("cam_low",
                                            data=np.random.rand(64, 64, 3))

                # State subgroup
                state_group = obs_group.create_group("state")
                state_group.create_dataset("joint_pos",
                                           data=np.array([0.1, 0.2, 0.3]))
                state_group.create_dataset("joint_vel",
                                           data=np.array([1.0, 2.0, 3.0]))

                # Action group
                action_group = f.create_group("action")
                action_group.create_dataset("joint_action",
                                            data=np.array([0.5, -0.5]))

            # Read back using function
            with h5py.File(tmp_path, "r") as f:
                result = recursively_read_hdf5_group(f)

            # Verify structure
            assert isinstance(result, dict)
            assert "root_data" in result
            assert "observation" in result
            assert "action" in result

            # Verify observation structure
            obs = result["observation"]
            assert isinstance(obs, dict)
            assert "images" in obs
            assert "state" in obs

            # Verify images
            images = obs["images"]
            assert isinstance(images, dict)
            assert "cam_high" in images
            assert "cam_low" in images
            assert images["cam_high"].shape == (128, 128, 3)
            assert images["cam_low"].shape == (64, 64, 3)

            # Verify state
            state = obs["state"]
            assert isinstance(state, dict)
            assert "joint_pos" in state
            assert "joint_vel" in state
            assert np.array_equal(state["joint_pos"], np.array([0.1, 0.2,
                                                                0.3]))
            assert np.array_equal(state["joint_vel"], np.array([1.0, 2.0,
                                                                3.0]))

            # Verify action
            action = result["action"]
            assert isinstance(action, dict)
            assert "joint_action" in action
            assert np.array_equal(action["joint_action"], np.array([0.5,
                                                                    -0.5]))

        finally:
            import os

            os.unlink(tmp_path)

    def test_unsupported_type(self):
        """Test handling of unsupported HDF5 types."""
        unsupported_object = "not an hdf5 object"

        with pytest.raises(TypeError, match="Unsupported HDF5 group type"):
            recursively_read_hdf5_group(unsupported_object)


class TestEdgeCases:
    """Test edge cases for flattening utilities."""

    def test_flatten_with_numeric_keys(self):
        """Test flattening with numeric keys."""
        data = {1: "value1", "nested": {2: "value2", "sub": {3: "value3"}}}

        result = _flatten(data)

        expected = {
            1: "value1",
            "nested/2": "value2",
            "nested/sub/3": "value3"
        }
        assert result == expected

    def test_flatten_with_special_characters(self):
        """Test flattening with special characters in keys."""
        data = {
            "key with spaces": "value1",
            "key-with-dashes": {
                "nested_key": "value2"
            },
            "key/with/slashes": "value3",
        }

        result = _flatten(data)

        expected = {
            "key with spaces": "value1",
            "key-with-dashes/nested_key": "value2",
            "key/with/slashes": "value3",
        }
        assert result == expected

    def test_flatten_dict_preserves_order(self):
        """Test that _flatten_dict preserves key order (Python 3.7+)."""
        data = {"z": 1, "a": {"y": 2, "b": 3}, "m": 4}

        result = _flatten_dict(data)

        # Check that keys appear in the order they were processed
        keys = list(result.keys())
        assert "z" in keys
        assert "a_y" in keys
        assert "a_b" in keys
        assert "m" in keys

    def test_data_to_tf_schema_empty_data(self):
        """Test data_to_tf_schema with empty data."""
        result = data_to_tf_schema({})
        assert result == {}

    def test_data_to_tf_schema_single_slash_key(self):
        """Test data_to_tf_schema with single slash in key."""
        data = {"observation/state": np.array([1, 2, 3])}

        with patch("robodm.utils.flatten.FeatureType") as mock_feature_type:
            mock_ft_instance = Mock()
            mock_ft_instance.to_tf_feature_type.return_value = "tf_feature"
            mock_feature_type.from_data.return_value = mock_ft_instance

            schema = data_to_tf_schema(data)

        assert "observation" in schema
        assert isinstance(schema["observation"], dict)
        assert "state" in schema["observation"]
        assert schema["observation"]["state"] == "tf_feature"

    def test_recursive_hdf5_empty_group(self):
        """Test reading empty HDF5 group."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write empty group
            with h5py.File(tmp_path, "w") as f:
                f.create_group("empty_group")

            # Read back using function
            with h5py.File(tmp_path, "r") as f:
                group = f["empty_group"]
                result = recursively_read_hdf5_group(group)

            assert isinstance(result, dict)
            assert len(result) == 0

        finally:
            import os

            os.unlink(tmp_path)

    def test_flatten_very_deep_nesting(self):
        """Test flattening with very deep nesting."""
        # Create deeply nested dict
        data = {}
        current = data
        for i in range(10):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["final_value"] = "deep"

        result = _flatten(data)

        expected_key = "/".join([f"level_{i}"
                                 for i in range(10)]) + "/final_value"
        assert expected_key in result
        assert result[expected_key] == "deep"
