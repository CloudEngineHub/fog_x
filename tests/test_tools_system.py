"""
Unit tests for the new tools system (registry, config, manager).
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock vllm module before importing our modules
sys.modules['vllm'] = Mock()

try:
    from PIL import Image
except ImportError:
    # Mock PIL if not available
    Image = Mock()

from robodm.agent.tools_registry import (
    ToolRegistry, VisionLanguageModel, analyze_image, analyze_trajectory,
    get_default_registry, register_user_tool
)
from robodm.agent.tools_config import (
    ToolsManager, create_vision_heavy_config, create_analysis_heavy_config,
    create_minimal_config, create_custom_config
)


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    def test_registry_init(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        
        # Should have default tools
        tools = registry.list_tools()
        assert "robo2vlm" in tools
        assert "analyze_image" in tools
        assert "analyze_trajectory" in tools
    
    def test_register_custom_tool(self):
        """Test registering custom tool."""
        registry = ToolRegistry()
        
        def custom_tool(x, y, multiplier=2):
            return (x + y) * multiplier
        
        registry.register_tool(
            name="custom_add",
            tool_func=custom_tool,
            description="Custom addition tool",
            signature="custom_add(x, y, multiplier=2) -> int",
            examples=["custom_add(2, 3)", "custom_add(1, 4, multiplier=3)"],
            default_params={"multiplier": 2}
        )
        
        assert "custom_add" in registry.list_tools()
        
        # Test tool usage
        tool = registry.get_tool("custom_add")
        assert tool(2, 3) == 10  # (2+3)*2
        
        # Test with custom params
        tool_custom = registry.get_tool("custom_add", {"multiplier": 5})
        assert tool_custom(2, 3) == 25  # (2+3)*5
    
    def test_tool_enable_disable(self):
        """Test enabling/disabling tools."""
        registry = ToolRegistry()
        
        # Disable a tool
        registry.disable_tool("robo2vlm")
        enabled_tools = registry.list_tools(enabled_only=True)
        all_tools = registry.list_tools(enabled_only=False)
        
        assert "robo2vlm" not in enabled_tools
        assert "robo2vlm" in all_tools
        
        # Re-enable the tool
        registry.enable_tool("robo2vlm")
        enabled_tools = registry.list_tools(enabled_only=True)
        assert "robo2vlm" in enabled_tools
    
    def test_tools_prompt_generation(self):
        """Test tools prompt generation."""
        registry = ToolRegistry()
        prompt = registry.get_tools_prompt()
        
        assert "Available Tools:" in prompt
        assert "robo2vlm" in prompt
        assert "Description:" in prompt
        assert "Signature:" in prompt
        assert "Usage examples:" in prompt
    
    def test_tools_namespace_creation(self):
        """Test tools namespace creation."""
        registry = ToolRegistry()
        
        config = {
            "analyze_image": {"blur_threshold": 50.0}
        }
        
        namespace = registry.create_tools_namespace(config)
        
        assert "robo2vlm" in namespace
        assert "analyze_image" in namespace
        assert callable(namespace["analyze_image"])


class TestAnalyzeImage:
    """Test cases for analyze_image tool."""
    
    def test_blur_detection(self):
        """Test blur detection functionality."""
        # Create sharp image (high frequency content)
        sharp_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        sharp_image[::2, ::2] = 255  # Create checkerboard pattern
        sharp_image[1::2, 1::2] = 0
        
        result = analyze_image(sharp_image, "blur", blur_threshold=50.0)
        
        assert "blur" in result
        assert "is_blurry" in result["blur"]
        assert "laplacian_variance" in result["blur"]
    
    def test_brightness_analysis(self):
        """Test brightness analysis."""
        # Create dark image
        dark_image = np.ones((64, 64, 3), dtype=np.uint8) * 50
        
        result = analyze_image(dark_image, "brightness", brightness_threshold=0.3)
        
        assert "brightness" in result
        assert "is_dark" in result["brightness"]
        assert "mean_brightness" in result["brightness"]
        assert result["brightness"]["is_dark"] == True
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        result = analyze_image(test_image, "features")
        
        assert "features" in result
        assert "shape" in result["features"]
        assert "mean_rgb" in result["features"]
        assert "std_rgb" in result["features"]
        assert result["features"]["shape"] == [32, 32, 3]
    
    def test_all_analysis(self):
        """Test running all analyses."""
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        result = analyze_image(test_image, "all")
        
        assert "blur" in result
        assert "brightness" in result
        assert "features" in result
    
    def test_error_handling(self):
        """Test error handling."""
        invalid_image = "not_an_array"
        
        result = analyze_image(invalid_image, "blur")
        
        assert "error" in result
        assert "Error in analyze_image" in result["error"]


class TestAnalyzeTrajectory:
    """Test cases for analyze_trajectory tool."""
    
    def test_velocity_computation(self):
        """Test velocity computation."""
        # Simple trajectory: linear motion
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
        
        velocities = analyze_trajectory(positions, "velocity")
        
        assert isinstance(velocities, np.ndarray)
        assert velocities.shape == (3, 2)  # N-1 velocity vectors
        assert np.allclose(velocities, [[1, 1], [1, 1], [1, 1]])
    
    def test_statistics_computation(self):
        """Test statistics computation."""
        trajectory_data = np.random.randn(50, 3)
        
        stats = analyze_trajectory(trajectory_data, "statistics", min_length=10)
        
        assert "length" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "is_long_enough" in stats
        assert stats["length"] == 50
        assert stats["is_long_enough"] == True
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create data with outliers
        normal_data = np.random.randn(100, 2)
        normal_data[50] = [10, 10]  # Add outlier
        
        anomalies = analyze_trajectory(normal_data, "anomalies", anomaly_threshold=2.0)
        
        assert "anomaly_indices" in anomalies
        assert "anomaly_count" in anomalies
        assert "anomaly_ratio" in anomalies
        assert 50 in anomalies["anomaly_indices"]  # Should detect the outlier
    
    def test_smoothing(self):
        """Test trajectory smoothing."""
        # Create noisy data
        t = np.linspace(0, 10, 50)
        clean_signal = np.sin(t)
        noisy_signal = clean_signal + 0.1 * np.random.randn(50)
        trajectory_2d = np.column_stack([t, noisy_signal])
        
        smoothed = analyze_trajectory(trajectory_2d, "smooth")
        
        assert isinstance(smoothed, np.ndarray)
        assert smoothed.shape == trajectory_2d.shape
        # Smoothed data should have lower variance
        assert np.var(smoothed[:, 1]) < np.var(trajectory_2d[:, 1])


class TestVisionLanguageModel:
    """Test cases for VisionLanguageModel tool."""
    
    @patch('robodm.agent.tools_registry.LLM')
    def test_vlm_initialization(self, mock_llm_class):
        """Test VLM initialization."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        vlm = VisionLanguageModel(model="test-model", temperature=0.2)
        
        assert vlm.model == "test-model"
        assert vlm.temperature == 0.2
    
    @patch('robodm.agent.tools_registry.LLM')
    def test_vlm_call(self, mock_llm_class):
        """Test VLM call functionality."""
        # Mock LLM response
        mock_llm = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Test response"
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm
        
        vlm = VisionLanguageModel()
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        result = vlm(test_image, "Test prompt")
        
        assert result == "Test response"
        mock_llm.generate.assert_called_once()
    
    def test_image_to_base64(self):
        """Test image to base64 conversion."""
        vlm = VisionLanguageModel()
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        b64_result = vlm._image_to_base64(test_image)
        
        assert isinstance(b64_result, str)
        assert len(b64_result) > 0


class TestToolsManager:
    """Test cases for ToolsManager."""
    
    def test_manager_initialization(self):
        """Test ToolsManager initialization."""
        config = {
            "enabled_tools": ["robo2vlm", "analyze_image"],
            "tool_params": {
                "analyze_image": {"blur_threshold": 75.0}
            }
        }
        
        manager = ToolsManager(config)
        
        enabled_tools = manager.list_tools()
        assert "robo2vlm" in enabled_tools
        assert "analyze_image" in enabled_tools
        assert "analyze_trajectory" not in enabled_tools  # Should be disabled
    
    def test_tool_configuration(self):
        """Test tool parameter configuration."""
        manager = ToolsManager()
        
        # Configure a tool
        manager.configure_tool("analyze_image", {"blur_threshold": 200.0})
        
        config = manager.get_config()
        assert "tool_params" in config
        assert "analyze_image" in config["tool_params"]
        assert config["tool_params"]["analyze_image"]["blur_threshold"] == 200.0
    
    def test_enable_disable_tools(self):
        """Test enabling and disabling tools."""
        manager = ToolsManager()
        
        # Disable a tool
        manager.disable_tool("analyze_trajectory")
        enabled_tools = manager.list_tools()
        assert "analyze_trajectory" not in enabled_tools
        
        # Re-enable the tool
        manager.enable_tool("analyze_trajectory")
        enabled_tools = manager.list_tools()
        assert "analyze_trajectory" in enabled_tools
    
    def test_tools_namespace(self):
        """Test tools namespace creation."""
        config = {
            "tool_params": {
                "analyze_image": {"blur_threshold": 150.0}
            }
        }
        
        manager = ToolsManager(config)
        namespace = manager.get_tools_namespace()
        
        assert "robo2vlm" in namespace
        assert "analyze_image" in namespace
        assert callable(namespace["analyze_image"])
    
    def test_tools_prompt(self):
        """Test tools prompt generation."""
        manager = ToolsManager()
        prompt = manager.get_tools_prompt()
        
        assert "Available Tools:" in prompt
        assert "robo2vlm" in prompt
        assert "analyze_image" in prompt
    
    def test_config_update(self):
        """Test configuration updates."""
        manager = ToolsManager()
        
        new_config = {
            "disabled_tools": ["analyze_trajectory"],
            "tool_params": {
                "robo2vlm": {"temperature": 0.05}
            }
        }
        
        manager.update_config(new_config)
        
        enabled_tools = manager.list_tools()
        assert "analyze_trajectory" not in enabled_tools
        
        config = manager.get_config()
        assert "robo2vlm" in config["tool_params"]
        assert config["tool_params"]["robo2vlm"]["temperature"] == 0.05


class TestConfigurationHelpers:
    """Test cases for configuration helper functions."""
    
    def test_vision_heavy_config(self):
        """Test vision-heavy configuration."""
        config = create_vision_heavy_config()
        
        assert "enabled_tools" in config
        assert "robo2vlm" in config["enabled_tools"]
        assert "analyze_image" in config["enabled_tools"]
        assert "tool_params" in config
    
    def test_analysis_heavy_config(self):
        """Test analysis-heavy configuration."""
        config = create_analysis_heavy_config()
        
        assert "enabled_tools" in config
        assert "analyze_trajectory" in config["enabled_tools"]
        assert "tool_params" in config
    
    def test_minimal_config(self):
        """Test minimal configuration."""
        config = create_minimal_config()
        
        assert "enabled_tools" in config
        assert len(config["enabled_tools"]) == 1
        assert "robo2vlm" in config["enabled_tools"]
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = create_custom_config(
            enabled_tools=["robo2vlm"],
            tool_params={"robo2vlm": {"temperature": 0.0}}
        )
        
        assert config["enabled_tools"] == ["robo2vlm"]
        assert config["tool_params"]["robo2vlm"]["temperature"] == 0.0


class TestUserToolRegistration:
    """Test cases for user tool registration."""
    
    def test_register_user_tool(self):
        """Test registering user-defined tool."""
        def my_custom_tool(data, threshold=0.5):
            """Custom tool for testing."""
            return np.mean(data) > threshold
        
        # Register the tool
        register_user_tool(
            name="custom_threshold",
            tool_func=my_custom_tool,
            description="Check if data mean exceeds threshold",
            signature="custom_threshold(data: np.ndarray, threshold: float = 0.5) -> bool",
            examples=["custom_threshold(trajectory_data)", "custom_threshold(values, threshold=0.8)"],
            default_params={"threshold": 0.5}
        )
        
        # Test that it's registered
        registry = get_default_registry()
        assert "custom_threshold" in registry.list_tools()
        
        # Test tool usage
        tool = registry.get_tool("custom_threshold")
        test_data = np.array([0.6, 0.7, 0.8])
        assert tool(test_data) == True  # Mean 0.7 > 0.5
        
        # Test with custom threshold
        tool_custom = registry.get_tool("custom_threshold", {"threshold": 0.8})
        assert tool_custom(test_data) == False  # Mean 0.7 < 0.8
    
    def test_tool_class_registration(self):
        """Test registering tool as a class."""
        class CustomAnalyzer:
            def __init__(self, sensitivity=1.0):
                self.sensitivity = sensitivity
            
            def __call__(self, data):
                return np.std(data) * self.sensitivity
        
        register_user_tool(
            name="custom_analyzer",
            tool_func=CustomAnalyzer,
            description="Custom data analyzer",
            signature="custom_analyzer(data: np.ndarray) -> float",
            examples=["custom_analyzer(sensor_data)"],
            default_params={"sensitivity": 1.0}
        )
        
        registry = get_default_registry()
        tool = registry.get_tool("custom_analyzer")
        
        test_data = np.array([1, 2, 3, 4, 5])
        result = tool(test_data)
        
        assert isinstance(result, (float, np.floating))
        assert result > 0


class TestIntegration:
    """Integration tests for the tools system."""
    
    def test_end_to_end_tool_usage(self):
        """Test end-to-end tool usage flow."""
        # Create configuration
        config = create_custom_config(
            enabled_tools=["analyze_image", "analyze_trajectory"],
            tool_params={
                "analyze_image": {"blur_threshold": 120.0},
                "analyze_trajectory": {"anomaly_threshold": 2.5}
            }
        )
        
        # Create manager
        manager = ToolsManager(config)
        
        # Get tools namespace
        tools = manager.get_tools_namespace()
        
        # Test image analysis tool
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image_result = tools["analyze_image"](test_image, "blur")
        
        assert "blur" in image_result
        assert "is_blurry" in image_result["blur"]
        
        # Test trajectory analysis tool
        test_trajectory = np.random.randn(50, 3)
        traj_result = tools["analyze_trajectory"](test_trajectory, "statistics")
        
        assert "length" in traj_result
        assert traj_result["length"] == 50
    
    def test_tool_configuration_persistence(self):
        """Test that tool configurations persist correctly."""
        config = {
            "tool_params": {
                "analyze_image": {"blur_threshold": 88.0}
            }
        }
        
        manager = ToolsManager(config)
        
        # Get tool and verify configuration
        tools = manager.get_tools_namespace()
        test_image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        
        result = tools["analyze_image"](test_image, "blur")
        
        # The threshold should be applied
        assert result["blur"]["threshold"] == 88.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])