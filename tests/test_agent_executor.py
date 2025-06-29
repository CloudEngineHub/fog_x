"""
Unit tests for robodm.agent.executor module.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import ray
from ray.data import Dataset

from robodm.agent.executor import Executor
from robodm.agent.tools import ToolsManager


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    return {
        "observation/image":
        np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),
        "observation/state":
        np.random.randn(10, 7),
        "action":
        np.random.randn(10, 3),
        "metadata": {
            "episode_id": 1,
            "scene": "kitchen"
        },
    }


@pytest.fixture
def sample_trajectories(sample_trajectory):
    """Create multiple sample trajectories for testing."""
    trajectories = []
    for i in range(5):
        traj = sample_trajectory.copy()
        traj["metadata"] = {
            "episode_id": i,
            "scene": "kitchen" if i < 3 else "office"
        }
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def mock_ray_dataset(sample_trajectories):
    """Create a mock Ray dataset for testing."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    dataset = ray.data.from_items(sample_trajectories)
    return dataset


class TestExecutorInit:
    """Test cases for Executor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)
        assert executor.max_retries == 3
        assert executor.tools_manager == tools_manager

    def test_custom_init(self):
        """Test custom initialization."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=5)
        assert executor.max_retries == 5
        assert executor.tools_manager == tools_manager

    def test_repr(self):
        """Test string representation."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=2)
        repr_str = repr(executor)
        assert "Executor" in repr_str
        assert "max_retries=2" in repr_str


class TestFunctionValidation:
    """Test cases for function validation."""

    def test_validate_filter_function_valid(self):
        """Test validation of valid filter function."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def valid_filter(trajectory: Dict[str, Any]) -> bool:
            return True

        assert executor.validate_function(valid_filter, "filter")

    def test_validate_filter_function_invalid_params(self):
        """Test validation of filter function with wrong parameters."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def invalid_filter() -> bool:
            return True

        assert not executor.validate_function(invalid_filter, "filter")

    def test_validate_map_function_valid(self):
        """Test validation of valid map function."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def valid_map(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            return trajectory

        assert executor.validate_function(valid_map, "map")

    def test_validate_aggregation_function_valid(self):
        """Test validation of valid aggregation function."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def valid_agg(trajectories: List[Dict[str, Any]]) -> Any:
            return len(trajectories)

        assert executor.validate_function(valid_agg, "aggregation")

    def test_validate_analysis_function_valid(self):
        """Test validation of valid analysis function."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def valid_analysis(trajectories: List[Dict[str, Any]]) -> str:
            return "analysis result"

        assert executor.validate_function(valid_analysis, "analysis")

    def test_validate_function_exception(self):
        """Test function validation with exception."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Function that can't be inspected
        invalid_func = "not_a_function"

        assert not executor.validate_function(invalid_func, "filter")


class TestSafeExecution:
    """Test cases for safe execution."""

    def test_safe_execute_success(self):
        """Test successful execution."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def success_func(x, y):
            return x + y

        result = executor.safe_execute(success_func, 2, 3)
        assert result == 5

    def test_safe_execute_failure(self):
        """Test execution with failure and retries."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=2)

        def fail_func():
            raise ValueError("Test error")

        result = executor.safe_execute(fail_func)
        assert isinstance(result, ValueError)
        assert str(result) == "Test error"

    def test_safe_execute_success_after_retry(self):
        """Test execution that succeeds after retries."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=3)

        call_count = 0

        def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry error")
            return "success"

        result = executor.safe_execute(retry_func)
        assert result == "success"
        assert call_count == 2


class TestCollectTrajectories:
    """Test cases for trajectory collection."""

    def test_collect_trajectories_small_dataset(self):
        """Test collecting trajectories from small dataset."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock small dataset
        mock_dataset = Mock()
        mock_dataset.count.return_value = 5
        mock_dataset.to_pandas.return_value = Mock()
        mock_dataset.to_pandas.return_value.iterrows.return_value = [
            (0, Mock(to_dict=lambda: {"traj": 1})),
            (1, Mock(to_dict=lambda: {"traj": 2})),
        ]

        trajectories = executor._collect_trajectories(mock_dataset)

        assert len(trajectories) == 2
        assert trajectories[0] == {"traj": 1}
        assert trajectories[1] == {"traj": 2}

    def test_collect_trajectories_large_dataset(self):
        """Test collecting trajectories from large dataset with sampling."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock large dataset
        mock_dataset = Mock()
        mock_dataset.count.return_value = 20000  # Larger than max_trajectories
        mock_sampled_dataset = Mock()
        mock_sampled_dataset.to_pandas.return_value = Mock()
        mock_sampled_dataset.to_pandas.return_value.iterrows.return_value = [
            (0, Mock(to_dict=lambda: {"sampled": True})),
        ]
        mock_dataset.random_sample.return_value = mock_sampled_dataset

        trajectories = executor._collect_trajectories(mock_dataset,
                                                      max_trajectories=100)

        assert len(trajectories) == 1
        assert trajectories[0] == {"sampled": True}
        mock_dataset.random_sample.assert_called_once()

    def test_collect_trajectories_fallback(self):
        """Test trajectory collection fallback to take()."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset that fails to_pandas but works with take
        mock_dataset = Mock()
        mock_dataset.count.return_value = 5
        mock_dataset.to_pandas.side_effect = Exception("Pandas failed")
        mock_dataset.take.return_value = [{"fallback": True}]

        trajectories = executor._collect_trajectories(mock_dataset)

        assert len(trajectories) == 1
        assert trajectories[0] == {"fallback": True}
        mock_dataset.take.assert_called_once_with(
            100)  # Default max_trajectories is 100

    def test_collect_trajectories_complete_failure(self):
        """Test trajectory collection complete failure."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset that fails everything
        mock_dataset = Mock()
        mock_dataset.count.return_value = 5
        mock_dataset.to_pandas.side_effect = Exception("Pandas failed")
        mock_dataset.take.side_effect = Exception("Take failed")

        with pytest.raises(RuntimeError,
                           match="Failed to collect trajectories"):
            executor._collect_trajectories(mock_dataset)


class TestApplyFilter:
    """Test cases for filter application."""

    def test_apply_filter_success(self, mock_ray_dataset):
        """Test successful filter application."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def simple_filter(trajectory: Dict[str, Any]) -> bool:
            return trajectory.get("metadata", {}).get("scene") == "kitchen"

        # This should work with the real Ray dataset
        filtered_dataset = executor.apply_filter(mock_ray_dataset,
                                                 simple_filter)

        # Check that we get a dataset back
        assert isinstance(filtered_dataset, Dataset)

        # Count should be <= original count
        original_count = mock_ray_dataset.count()
        filtered_count = filtered_dataset.count()
        assert filtered_count <= original_count

    def test_apply_filter_with_exception_in_filter(self):
        """Test filter application when filter function raises exception."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset operations
        mock_dataset = Mock()
        mock_filtered_dataset = Mock()
        mock_final_dataset = Mock()

        # Set up the chain of mock calls
        mock_dataset.map_batches.return_value = mock_filtered_dataset
        mock_filtered_dataset.filter.return_value = mock_final_dataset
        mock_final_dataset.map_batches.return_value = mock_final_dataset

        def failing_filter(trajectory: Dict[str, Any]) -> bool:
            raise ValueError("Filter failed")

        # Should not raise exception, but handle it gracefully
        result = executor.apply_filter(mock_dataset, failing_filter)
        assert result == mock_final_dataset

    def test_apply_filter_ray_failure(self):
        """Test filter application when Ray operations fail."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset that fails map_batches
        mock_dataset = Mock()
        mock_dataset.map_batches.side_effect = Exception("Ray failed")

        def simple_filter(trajectory: Dict[str, Any]) -> bool:
            return True

        with pytest.raises(RuntimeError, match="Failed to apply filter"):
            executor.apply_filter(mock_dataset, simple_filter)


class TestApplyMap:
    """Test cases for map application."""

    def test_apply_map_success(self, mock_ray_dataset):
        """Test successful map application."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        def simple_map(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            result = trajectory.copy()
            result["new_field"] = "added"
            return result

        # This should work with the real Ray dataset
        mapped_dataset = executor.apply_map(mock_ray_dataset, simple_map)

        # Check that we get a dataset back
        assert isinstance(mapped_dataset, Dataset)

    def test_apply_map_with_exception(self):
        """Test map application when map function raises exception."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset
        mock_dataset = Mock()
        mock_mapped_dataset = Mock()
        mock_dataset.map_batches.return_value = mock_mapped_dataset

        def failing_map(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Map failed")

        # Should not raise exception, but handle it gracefully
        result = executor.apply_map(mock_dataset, failing_map)
        assert result == mock_mapped_dataset

    def test_apply_map_ray_failure(self):
        """Test map application when Ray operations fail."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock dataset that fails map_batches
        mock_dataset = Mock()
        mock_dataset.map_batches.side_effect = Exception("Ray failed")

        def simple_map(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            return trajectory

        with pytest.raises(RuntimeError, match="Failed to apply map"):
            executor.apply_map(mock_dataset, simple_map)


class TestApplyAggregation:
    """Test cases for aggregation application."""

    def test_apply_aggregation_success(self):
        """Test successful aggregation application."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock the _collect_trajectories method
        trajectories = [
            {
                "metadata": {
                    "scene": "kitchen"
                }
            },
            {
                "metadata": {
                    "scene": "office"
                }
            },
            {
                "metadata": {
                    "scene": "kitchen"
                }
            },
        ]

        with patch.object(executor,
                          "_collect_trajectories",
                          return_value=trajectories):
            mock_dataset = Mock()

            def count_by_scene(trajs: List[Dict[str, Any]]) -> Dict[str, int]:
                from collections import Counter

                scenes = [
                    t.get("metadata", {}).get("scene", "unknown")
                    for t in trajs
                ]
                return dict(Counter(scenes))

            result = executor.apply_aggregation(mock_dataset, count_by_scene)

            assert result == {"kitchen": 2, "office": 1}

    def test_apply_aggregation_failure(self):
        """Test aggregation application failure."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock _collect_trajectories to raise exception
        with patch.object(
                executor,
                "_collect_trajectories",
                side_effect=Exception("Collection failed"),
        ):
            mock_dataset = Mock()

            def simple_agg(trajs: List[Dict[str, Any]]) -> int:
                return len(trajs)

            with pytest.raises(RuntimeError,
                               match="Failed to apply aggregation"):
                executor.apply_aggregation(mock_dataset, simple_agg)


class TestApplyAnalysis:
    """Test cases for analysis application."""

    def test_apply_analysis_success(self):
        """Test successful analysis application."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock the _collect_trajectories method
        trajectories = [
            {
                "observation/image": np.random.rand(10, 64, 64, 3)
            },
            {
                "observation/image": np.random.rand(15, 64, 64, 3)
            },
        ]

        with patch.object(executor,
                          "_collect_trajectories",
                          return_value=trajectories):
            mock_dataset = Mock()

            def analyze_lengths(trajs: List[Dict[str, Any]]) -> str:
                lengths = [len(t["observation/image"]) for t in trajs]
                avg_length = sum(lengths) / len(lengths)
                return f"Average length: {avg_length:.1f}"

            result = executor.apply_analysis(mock_dataset, analyze_lengths)

            assert result == "Average length: 12.5"

    def test_apply_analysis_failure(self):
        """Test analysis application failure."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Mock _collect_trajectories to raise exception
        with patch.object(
                executor,
                "_collect_trajectories",
                side_effect=Exception("Collection failed"),
        ):
            mock_dataset = Mock()

            def simple_analysis(trajs: List[Dict[str, Any]]) -> str:
                return "analysis"

            with pytest.raises(RuntimeError, match="Failed to apply analysis"):
                executor.apply_analysis(mock_dataset, simple_analysis)


class TestGetExecutionStats:
    """Test cases for execution statistics."""

    @patch("ray.is_initialized")
    @patch("ray.cluster_resources")
    def test_get_execution_stats_ray_initialized(self, mock_cluster_resources,
                                                 mock_ray_init):
        """Test execution stats when Ray is initialized."""
        mock_ray_init.return_value = True
        mock_cluster_resources.return_value = {"CPU": 4, "memory": 8000000000}

        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=5)
        stats = executor.get_execution_stats()

        assert stats["max_retries"] == 5
        assert stats["ray_cluster_resources"]["CPU"] == 4

    @patch("ray.is_initialized")
    def test_get_execution_stats_ray_not_initialized(self, mock_ray_init):
        """Test execution stats when Ray is not initialized."""
        mock_ray_init.return_value = False

        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)
        stats = executor.get_execution_stats()

        assert stats["max_retries"] == 3
        assert stats["ray_cluster_resources"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
