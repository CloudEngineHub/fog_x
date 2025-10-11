"""
Unit tests for robodm.agent.tools module.
"""

import base64
import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from robodm.agent.tools import (  # Legacy compatibility functions; New tool system
    ImageAnalysisTool, ToolsManager, TrajectoryAnalysisTool,
    VisionLanguageModelTool, analyze_image, analyze_trajectory, create_manager,
    detect_scene_changes, extract_keyframes, get_registry)


class TestToolsManager:
    """Test cases for the new ToolsManager system."""

    def test_tools_manager_init(self):
        """Test ToolsManager initialization."""
        manager = ToolsManager()

        # Should have tools registered
        tools = manager.list_tools()
        assert len(tools) > 0

        # Check that essential tools are available
        assert "robo2vlm" in tools
        # Other tools may not be available due to import mocking in test environment
        # This is acceptable as long as basic functionality works

    def test_tools_manager_with_config(self):
        """Test ToolsManager with configuration."""
        config = {
            "tools": {
                "robo2vlm": {
                    "temperature": 0.05,
                    "max_tokens": 512
                }
            },
            "disabled_tools": ["analyze_trajectory"],
        }

        manager = ToolsManager(config=config)
        enabled_tools = manager.list_tools(enabled_only=True)

        # Should not include disabled tool
        assert "analyze_trajectory" not in enabled_tools
        assert "robo2vlm" in enabled_tools

    def test_get_tool_instance(self):
        """Test getting tool instances."""
        manager = ToolsManager()

        # Get VLM tool
        vlm_tool = manager.get_tool("robo2vlm")
        assert vlm_tool is not None
        assert hasattr(vlm_tool, "__call__")

        # Get image analysis tool
        img_tool = manager.get_tool("analyze_image")
        assert img_tool is not None
        assert hasattr(img_tool, "__call__")

    def test_tools_namespace(self):
        """Test getting tools namespace for code execution."""
        manager = ToolsManager()
        namespace = manager.get_tools_namespace()

        assert isinstance(namespace, dict)
        assert "robo2vlm" in namespace
        # Note: Other tools may not be available due to test environment mocking

        # Test that available tools are callable
        for tool in namespace.values():
            assert hasattr(tool, "__call__")


class TestImageAnalysisTool:
    """Test cases for ImageAnalysisTool."""

    def test_image_analysis_tool_init(self):
        """Test ImageAnalysisTool initialization."""
        tool = ImageAnalysisTool(blur_threshold=80.0,
                                 brightness_threshold=0.25)

        assert tool.blur_threshold == 80.0
        assert tool.brightness_threshold == 0.25
        assert tool.enabled is True

    def test_image_analysis_all_operations(self):
        """Test image analysis with all operations."""
        tool = ImageAnalysisTool()

        # Test with RGB image
        rgb_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = tool(rgb_image, "all")

        assert isinstance(result, dict)
        assert "blur" in result
        assert "brightness" in result
        assert "features" in result

        # Check blur analysis
        assert "is_blurry" in result["blur"]
        assert "laplacian_variance" in result["blur"]
        assert "threshold" in result["blur"]

        # Check brightness analysis
        assert "mean_brightness" in result["brightness"]
        assert "is_dark" in result["brightness"]
        assert "is_bright" in result["brightness"]

        # Check features
        assert "shape" in result["features"]
        assert "mean_rgb" in result["features"]

    def test_image_analysis_specific_operations(self):
        """Test image analysis with specific operations."""
        tool = ImageAnalysisTool()

        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        # Test blur only
        blur_result = tool(image, "blur")
        assert "blur" in blur_result
        assert "brightness" not in blur_result

        # Test brightness only
        brightness_result = tool(image, "brightness")
        assert "brightness" in brightness_result
        assert "blur" not in brightness_result

    def test_image_analysis_legacy_function(self):
        """Test legacy analyze_image function."""
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        result = analyze_image(image, "all")

        assert isinstance(result, dict)
        assert "blur" in result or "brightness" in result or "features" in result


class TestVisionLanguageModelTool:
    """Test cases for VisionLanguageModelTool."""

    @patch("robodm.agent.tools.implementations.LLM")
    def test_vlm_tool_init(self, mock_llm_class):
        """Test VisionLanguageModelTool initialization."""
        tool = VisionLanguageModelTool(model="test-model", temperature=0.05)

        assert tool.model == "test-model"
        assert tool.temperature == 0.05
        assert tool.enabled is True

    @patch("robodm.agent.tools.implementations.LLM")
    def test_vlm_tool_call(self, mock_llm_class):
        """Test VisionLanguageModelTool call."""
        # Mock VLM and response
        mock_vlm = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Yes, there is occlusion in the image."
        mock_vlm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_vlm

        tool = VisionLanguageModelTool()

        # Test data
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        prompt = "Is there occlusion in this image?"

        result = tool(frame, prompt)

        assert result == "Yes, there is occlusion in the image."
        mock_vlm.generate.assert_called_once()

        # Check that the generated call includes image and text
        call_args = mock_vlm.generate.call_args
        multimodal_prompt = call_args[0][0][0]  # First prompt in the list

        assert len(multimodal_prompt) == 2  # image and text components
        assert multimodal_prompt[0]["type"] == "image_url"
        assert multimodal_prompt[1]["type"] == "text"
        assert multimodal_prompt[1]["text"] == prompt

    @patch("robodm.agent.tools.implementations.LLM")
    def test_vlm_tool_error_handling(self, mock_llm_class):
        """Test VisionLanguageModelTool error handling."""
        # Mock VLM to raise exception
        mock_vlm = Mock()
        mock_vlm.generate.side_effect = RuntimeError("VLM failed")
        mock_llm_class.return_value = mock_vlm

        tool = VisionLanguageModelTool()

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        prompt = "test prompt"

        result = tool(frame, prompt)

        assert "Error in robo2vlm" in result
        assert "VLM failed" in result

    def test_vlm_tool_metadata(self):
        """Test VisionLanguageModelTool metadata."""
        metadata = VisionLanguageModelTool.get_metadata()

        assert metadata.name == "robo2vlm"
        assert "vision-language model" in metadata.description.lower()
        assert len(metadata.examples) > 0
        assert "vision" in metadata.tags

    def test_vlm_tool_validation(self):
        """Test VisionLanguageModelTool configuration validation."""
        # Valid configuration
        tool = VisionLanguageModelTool(temperature=0.1, max_tokens=256)
        # Should not raise exception

        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between"):
            VisionLanguageModelTool(temperature=3.0)

        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            VisionLanguageModelTool(max_tokens=-1)


class TestTrajectoryAnalysisTool:
    """Test cases for TrajectoryAnalysisTool."""

    def test_trajectory_tool_init(self):
        """Test TrajectoryAnalysisTool initialization."""
        tool = TrajectoryAnalysisTool(anomaly_threshold=2.5,
                                      min_length=15,
                                      smoothing_window=7)

        assert tool.anomaly_threshold == 2.5
        assert tool.min_length == 15
        assert tool.smoothing_window == 7
        assert tool.enabled is True

    def test_trajectory_statistics(self):
        """Test trajectory statistics computation."""
        tool = TrajectoryAnalysisTool()

        # Test data
        data = np.random.randn(20, 6)  # 20 timesteps, 6 joints

        result = tool(data, "statistics")

        assert isinstance(result, dict)
        assert "length" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "is_long_enough" in result

        assert result["length"] == 20
        assert len(result["mean"]) == 6  # 6 joints

    def test_trajectory_velocity(self):
        """Test trajectory velocity computation."""
        tool = TrajectoryAnalysisTool()

        # Simple position data
        data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])  # Linear motion

        velocity = tool(data, "velocity")

        assert isinstance(velocity, np.ndarray)
        assert velocity.shape == (3, 2)  # N-1 timesteps
        # Should be constant velocity of [1, 1]
        assert np.allclose(velocity, [[1, 1], [1, 1], [1, 1]])

    def test_trajectory_anomaly_detection(self):
        """Test trajectory anomaly detection."""
        tool = TrajectoryAnalysisTool(anomaly_threshold=2.0)

        # Create data with clear anomaly
        normal_data = np.random.randn(50, 3) * 0.1  # Small variance
        anomaly_point = np.array([[10, 10, 10]])  # Clear outlier

        data = np.vstack([normal_data[:25], anomaly_point, normal_data[25:]])

        result = tool(data, "anomalies")

        assert isinstance(result, dict)
        assert "anomaly_indices" in result
        assert "anomaly_count" in result
        assert "anomaly_ratio" in result

        # Should detect the anomaly at index 25
        assert 25 in result["anomaly_indices"]
        assert result["anomaly_count"] >= 1

    def test_trajectory_smoothing(self):
        """Test trajectory smoothing."""
        tool = TrajectoryAnalysisTool(smoothing_window=3)

        # Noisy signal
        t = np.linspace(0, 1, 20)
        clean_signal = np.sin(2 * np.pi * t)
        noisy_signal = clean_signal + 0.1 * np.random.randn(20)
        data = noisy_signal.reshape(-1, 1)

        smoothed = tool(data, "smooth")

        assert isinstance(smoothed, np.ndarray)
        assert smoothed.shape == data.shape

        # Smoothed signal should have less variance
        assert np.var(smoothed) <= np.var(data)

    def test_trajectory_tool_metadata(self):
        """Test TrajectoryAnalysisTool metadata."""
        metadata = TrajectoryAnalysisTool.get_metadata()

        assert metadata.name == "analyze_trajectory"
        assert "trajectory" in metadata.description.lower()
        assert len(metadata.examples) > 0
        assert "trajectory" in metadata.tags

    def test_trajectory_legacy_function(self):
        """Test legacy analyze_trajectory function."""
        data = np.random.randn(15, 4)

        result = analyze_trajectory(data, "statistics")

        assert isinstance(result, dict)
        assert "length" in result
        assert result["length"] == 15


class TestTrajectoryUtilities:
    """Test cases for trajectory utility functions."""

    def test_extract_keyframes(self):
        """Test keyframe extraction."""
        # Create sequence of images
        images = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)

        indices, keyframes = extract_keyframes(images, num_keyframes=5)

        assert len(indices) == 5
        assert keyframes.shape == (5, 64, 64, 3)
        assert indices == [0, 4, 9, 14, 19]  # Uniform sampling

    def test_extract_keyframes_short_sequence(self):
        """Test keyframe extraction from short sequence."""
        images = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)

        indices, keyframes = extract_keyframes(images, num_keyframes=5)

        # Should return all frames when requested more than available
        assert len(indices) == 3
        assert keyframes.shape == (3, 32, 32, 3)

    def test_detect_scene_changes_with_vlm(self):
        """Test scene change detection using VLM tool."""
        # Test the utility function
        images = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)

        # Mock VLM function
        mock_vlm_func = Mock()

        # Mock VLM responses for scene change detection
        mock_vlm_func.side_effect = [
            "Kitchen scene with table",  # Frame 0 description
            "Kitchen scene with table",  # Frame 1 description (similar)
            "yes",  # Similarity check frame 1 (similar -> no change)
            "Living room with sofa",  # Frame 2 description (different)
            "no",  # Similarity check frame 2 (different -> change)
            "Living room with sofa",  # Frame 3 description (similar)
            "yes",  # Similarity check frame 3 (similar -> no change)
        ]

        scene_changes = detect_scene_changes(images, mock_vlm_func)

        assert len(scene_changes) == 1
        assert scene_changes[0] == 2  # Scene change at frame 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
