"""
Unit tests for the robodm.agent module.
"""

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock vllm module before importing our modules
sys.modules["vllm"] = Mock()

import ray
from ray.data import Dataset

from robodm.agent import Agent, Executor, Planner
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

    # Create a simple Ray dataset from list
    dataset = ray.data.from_items(sample_trajectories)
    return dataset


# Removed TestRobo2VLM class since robo2vlm is now part of the tools system


class TestPlanner:
    """Test cases for Planner class."""

    @patch("robodm.agent.planner.LLM")
    def test_planner_init(self, mock_llm_class):
        """Test Planner initialization."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        tools_manager = ToolsManager()
        planner = Planner(llm_model="test-model", tools_manager=tools_manager)

        assert planner.llm_model == "test-model"
        assert planner.llm == mock_llm
        assert planner.tools_manager == tools_manager
        mock_llm_class.assert_called_once_with(model="test-model")

    @patch("robodm.agent.planner.LLM")
    def test_generate_filter_function(self, mock_llm_class, mock_ray_dataset):
        """Test filter function generation with dynamic schema."""
        # Mock LLM response
        mock_llm = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = """    
    # Check for frame count using actual schema
    temporal_keys = [k for k in trajectory.keys() if hasattr(trajectory[k], 'shape') and len(trajectory[k].shape) >= 2]
    if temporal_keys:
        return len(trajectory[temporal_keys[0]]) > 5
    return False"""
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm

        tools_manager = ToolsManager()
        planner = Planner(tools_manager=tools_manager)
        filter_func = planner.generate_filter_function(
            "trajectories with more than 5 frames", dataset=mock_ray_dataset)

        # Test generated function
        sample_traj = {"observation/image": np.random.randn(10, 64, 64, 3)}
        result = filter_func(sample_traj)

        assert isinstance(result, bool)
        assert result is True  # 10 > 5

    def test_inspect_dataset_schema(self, sample_trajectories):
        """Test dataset schema inspection."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        dataset = ray.data.from_items(sample_trajectories)
        planner = Planner.__new__(Planner)  # Create without __init__
        planner._cached_schema = None

        schema_info = planner.inspect_dataset_schema(dataset)

        assert "keys" in schema_info
        assert "shapes" in schema_info
        assert "dtypes" in schema_info
        assert "image_keys" in schema_info
        assert "temporal_keys" in schema_info

        # Check that it found the expected keys
        assert "observation/image" in schema_info["keys"]
        assert "metadata" in schema_info["keys"]

        # Check image detection
        if "observation/image" in schema_info["image_keys"]:
            assert schema_info["has_images"] is True

    def test_generate_schema_prompt(self, sample_trajectories):
        """Test schema prompt generation."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        dataset = ray.data.from_items(sample_trajectories)
        planner = Planner.__new__(Planner)  # Create without __init__
        planner._cached_schema = None

        schema_info = planner.inspect_dataset_schema(dataset)
        schema_prompt = planner._generate_schema_prompt(schema_info)

        assert "Dataset Schema:" in schema_prompt
        assert "observation/image" in schema_prompt
        assert "shape" in schema_prompt.lower()

    def test_clean_generated_code(self):
        """Test code cleaning functionality."""
        planner = Planner.__new__(Planner)  # Create without __init__

        code = """if True:
    return True
else:
    return False"""

        cleaned = planner._clean_generated_code(code)
        lines = cleaned.split("\n")

        # Check that all lines are properly indented
        for line in lines:
            if line.strip():
                assert line.startswith("    ")


class TestExecutor:
    """Test cases for Executor class."""

    def test_executor_init(self):
        """Test Executor initialization."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=5)
        assert executor.max_retries == 5
        assert executor.tools_manager == tools_manager

    def test_validate_function(self):
        """Test function validation."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)

        # Valid filter function
        def valid_filter(trajectory: Dict[str, Any]) -> bool:
            return True

        assert executor.validate_function(valid_filter, "filter")

        # Invalid function (wrong parameter count)
        def invalid_filter() -> bool:
            return True

        assert not executor.validate_function(invalid_filter, "filter")

    def test_safe_execute(self):
        """Test safe execution with retries."""
        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager, max_retries=2)

        # Function that succeeds
        def success_func(x):
            return x * 2

        result = executor.safe_execute(success_func, 5)
        assert result == 10

        # Function that always fails
        def fail_func():
            raise ValueError("Test error")

        result = executor.safe_execute(fail_func)
        assert isinstance(result, ValueError)

    @patch("ray.is_initialized")
    def test_get_execution_stats(self, mock_ray_init):
        """Test execution statistics."""
        mock_ray_init.return_value = False

        tools_manager = ToolsManager()
        executor = Executor(tools_manager=tools_manager)
        stats = executor.get_execution_stats()

        assert "max_retries" in stats
        assert stats["max_retries"] == 3
        assert "ray_cluster_resources" in stats


class TestAgent:
    """Test cases for Agent class."""

    @patch("robodm.agent.agent.Planner")
    @patch("robodm.agent.agent.Executor")
    def test_agent_init(self, mock_executor_class, mock_planner_class,
                        mock_ray_dataset):
        """Test Agent initialization."""
        mock_planner = Mock()
        mock_executor = Mock()
        mock_planner_class.return_value = mock_planner
        mock_executor_class.return_value = mock_executor

        agent = Agent(mock_ray_dataset, llm_model="test-model")

        assert agent.dataset == mock_ray_dataset
        assert agent.planner == mock_planner
        assert agent.executor == mock_executor
        assert agent.tools_manager is not None
        mock_planner_class.assert_called_once_with(
            llm_model="test-model", tools_manager=agent.tools_manager)
        mock_executor_class.assert_called_once_with(
            tools_manager=agent.tools_manager)

    @patch("robodm.agent.agent.Planner")
    @patch("robodm.agent.agent.Executor")
    def test_agent_filter(self, mock_executor_class, mock_planner_class,
                          mock_ray_dataset):
        """Test Agent filter functionality."""
        # Mock planner and executor
        mock_planner = Mock()
        mock_executor = Mock()
        mock_filter_func = Mock(return_value=True)
        mock_filtered_dataset = Mock()

        mock_planner.generate_filter_function.return_value = mock_filter_func
        mock_executor.apply_filter.return_value = mock_filtered_dataset

        mock_planner_class.return_value = mock_planner
        mock_executor_class.return_value = mock_executor

        agent = Agent(mock_ray_dataset)
        result = agent.filter("trajectories with occlusion")

        assert result == mock_filtered_dataset
        mock_planner.generate_filter_function.assert_called_once_with(
            "trajectories with occlusion", dataset=mock_ray_dataset)
        mock_executor.apply_filter.assert_called_once_with(
            mock_ray_dataset, mock_filter_func)

    @patch("robodm.agent.agent.Planner")
    @patch("robodm.agent.agent.Executor")
    def test_agent_map(self, mock_executor_class, mock_planner_class,
                       mock_ray_dataset):
        """Test Agent map functionality."""
        # Mock planner and executor
        mock_planner = Mock()
        mock_executor = Mock()
        mock_map_func = Mock()
        mock_mapped_dataset = Mock()

        mock_planner.generate_map_function.return_value = mock_map_func
        mock_executor.apply_map.return_value = mock_mapped_dataset

        mock_planner_class.return_value = mock_planner
        mock_executor_class.return_value = mock_executor

        agent = Agent(mock_ray_dataset)
        result = agent.map("add frame differences")

        assert result == mock_mapped_dataset
        mock_planner.generate_map_function.assert_called_once_with(
            "add frame differences", dataset=mock_ray_dataset)
        mock_executor.apply_map.assert_called_once_with(
            mock_ray_dataset, mock_map_func)

    def test_agent_count(self, mock_ray_dataset):
        """Test Agent count functionality."""
        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset)
            count = agent.count()

        assert count == 5  # mock_ray_dataset has 5 trajectories
        assert isinstance(count, int)

    def test_agent_len(self, mock_ray_dataset):
        """Test Agent __len__ functionality."""
        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset)
            length = len(agent)

        assert length == 5  # mock_ray_dataset has 5 trajectories
        assert isinstance(length, int)

    def test_agent_repr(self, mock_ray_dataset):
        """Test Agent string representation."""
        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset)
            repr_str = repr(agent)

        assert "Agent" in repr_str
        assert "count=5" in repr_str

    def test_agent_inspect_schema(self, mock_ray_dataset):
        """Test Agent schema inspection."""
        with patch("robodm.agent.agent.Planner") as mock_planner_class:
            mock_planner = Mock()
            mock_schema_info = {
                "keys": ["observation/image", "action"],
                "shapes": {
                    "observation/image": [10, 64, 64, 3]
                },
                "dtypes": {
                    "observation/image": "uint8"
                },
                "has_images": True,
                "image_keys": ["observation/image"],
                "temporal_keys": ["observation/image", "action"],
                "scalar_keys": [],
            }
            mock_planner.inspect_dataset_schema.return_value = mock_schema_info
            mock_planner_class.return_value = mock_planner

            with patch("robodm.agent.agent.Executor"):
                agent = Agent(mock_ray_dataset)
                schema_info = agent.inspect_schema()

        assert schema_info == mock_schema_info
        mock_planner.inspect_dataset_schema.assert_called_once_with(
            mock_ray_dataset)

    def test_agent_with_tools_config(self, mock_ray_dataset):
        """Test Agent initialization with tools configuration."""
        tools_config = {
            "tools": {
                "robo2vlm": {
                    "temperature": 0.05,
                    "max_tokens": 512
                }
            },
            "disabled_tools": ["analyze_trajectory"],
        }

        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset, tools_config=tools_config)

            # Check that tools manager was configured
            assert agent.tools_manager is not None

            # Check that tools are available
            tools = agent.list_tools()
            assert "robo2vlm" in tools
            assert "analyze_trajectory" not in tools  # Should be disabled

    def test_agent_with_preset_config(self, mock_ray_dataset):
        """Test Agent initialization with preset configuration."""
        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset, tools_config="minimal")

            # Check that tools manager was configured with preset
            assert agent.tools_manager is not None

            # Minimal config should have limited tools
            tools = agent.list_tools()
            assert "robo2vlm" in tools

    def test_agent_tools_management(self, mock_ray_dataset):
        """Test Agent tools management functionality."""
        with patch("robodm.agent.agent.Planner"), patch(
                "robodm.agent.agent.Executor"):
            agent = Agent(mock_ray_dataset)

            # Test list tools
            tools = agent.list_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0

            # Test enable/disable tools
            if "analyze_image" in tools:
                agent.disable_tool("analyze_image")
                updated_tools = agent.list_tools()
                assert "analyze_image" not in updated_tools

                agent.enable_tool("analyze_image")
                updated_tools = agent.list_tools()
                assert "analyze_image" in updated_tools

            # Test get tools info
            info = agent.get_tools_info()
            assert isinstance(info, str)
            assert len(info) > 0

    def test_agent_describe_dataset(self, mock_ray_dataset):
        """Test Agent dataset description."""
        with patch("robodm.agent.agent.Planner") as mock_planner_class:
            mock_planner = Mock()
            mock_schema_info = {
                "keys": ["observation/image", "metadata"],
                "shapes": {
                    "observation/image": [10, 64, 64, 3]
                },
                "dtypes": {
                    "observation/image": "uint8"
                },
                "sample_values": {
                    "metadata": {
                        "scene": "kitchen"
                    }
                },
                "has_images": True,
                "image_keys": ["observation/image"],
                "temporal_keys": ["observation/image"],
                "scalar_keys": ["metadata"],
            }
            mock_planner.inspect_dataset_schema.return_value = mock_schema_info
            mock_planner_class.return_value = mock_planner

            with patch("robodm.agent.agent.Executor"):
                agent = Agent(mock_ray_dataset)
                description = agent.describe_dataset()

        assert "Dataset with 2 feature keys:" in description
        assert "observation/image" in description
        assert "image data" in description
        assert "metadata" in description


class TestIntegration:
    """Integration tests for the complete Agent system."""

    @pytest.mark.slow
    def test_end_to_end_filter_simple(self, sample_trajectories):
        """Test end-to-end filtering with simple logic."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create dataset
        dataset = ray.data.from_items(sample_trajectories)

        # Mock the LLM to return simple filter logic
        with patch("robodm.agent.planner.LLM") as mock_llm_class:
            mock_llm = Mock()
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = """
    # Filter trajectories from kitchen
    scene = trajectory.get("metadata", {}).get("scene", "")
    return scene == "kitchen" """
            mock_llm.generate.return_value = [mock_output]
            mock_llm_class.return_value = mock_llm

            # Create agent and apply filter
            agent = Agent(dataset)
            filtered_dataset = agent.filter("trajectories from kitchen")

            # Check results
            filtered_count = filtered_dataset.count()
            assert filtered_count == 3  # 3 kitchen trajectories in sample data

    def test_error_propagation(self, mock_ray_dataset):
        """Test error propagation through the system."""
        with patch("robodm.agent.agent.Planner") as mock_planner_class:
            mock_planner = Mock()
            mock_planner.generate_filter_function.side_effect = RuntimeError(
                "LLM failed")
            mock_planner_class.return_value = mock_planner

            with patch("robodm.agent.agent.Executor"):
                agent = Agent(mock_ray_dataset)

                with pytest.raises(RuntimeError, match="LLM failed"):
                    agent.filter("test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
