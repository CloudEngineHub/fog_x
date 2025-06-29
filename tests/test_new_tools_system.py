"""
Tests for the reorganized tools system.
"""

import pytest
import numpy as np
import sys

# Mock vllm module
sys.modules['vllm'] = type('MockVLLM', (), {
    'LLM': type('MockLLM', (), {
        '__init__': lambda self, model: None,
        'generate': lambda self, prompts, params: [type('MockOutput', (), {
            'outputs': [type('MockGeneration', (), {'text': 'Mock response'})()]
        })()]
    }),
    'SamplingParams': lambda **kwargs: None
})()

from robodm.agent.tools import (
    ToolsManager, 
    create_vision_config, 
    create_analysis_config,
    create_minimal_config,
    create_custom_config,
    analyze_image,
    analyze_trajectory
)
from robodm.agent.tools.manager import register_tool


class TestNewToolsSystem:
    """Test the reorganized tools system."""
    
    def test_tools_manager_initialization(self):
        """Test ToolsManager initialization."""
        manager = ToolsManager()
        
        # Should have default tools
        tools = manager.list_tools()
        assert "robo2vlm" in tools
        assert "analyze_image" in tools
        assert "analyze_trajectory" in tools
    
    def test_configuration_templates(self):
        """Test configuration templates."""
        vision_config = create_vision_config()
        analysis_config = create_analysis_config()
        minimal_config = create_minimal_config()
        
        assert "enabled_tools" in vision_config
        assert "robo2vlm" in vision_config["enabled_tools"]
        
        assert "enabled_tools" in analysis_config
        assert "analyze_trajectory" in analysis_config["enabled_tools"]
        
        assert "enabled_tools" in minimal_config
        assert len(minimal_config["enabled_tools"]) == 1
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = create_custom_config(
            enabled_tools=["analyze_image"],
            tool_params={"analyze_image": {"blur_threshold": 50.0}}
        )
        
        manager = ToolsManager(config)
        tools = manager.list_tools()
        
        assert "analyze_image" in tools
        assert "robo2vlm" not in tools  # Should be disabled
        assert "analyze_trajectory" not in tools  # Should be disabled
    
    def test_tool_registration(self):
        """Test tool registration."""
        def custom_tool(data, threshold=1.0):
            return np.mean(data) > threshold
        
        manager = ToolsManager()
        manager.register_tool(
            name="custom_threshold",
            implementation=custom_tool,
            description="Custom threshold tool",
            signature="custom_threshold(data, threshold=1.0) -> bool",
            examples=["custom_threshold(data)"],
            default_params={"threshold": 1.0}
        )
        
        tools = manager.list_tools()
        assert "custom_threshold" in tools
        
        # Test tool usage
        tool = manager.get_tool("custom_threshold")
        result = tool(np.array([2, 3, 4]))
        assert result == True  # Mean 3.0 > 1.0
    
    def test_tool_configuration(self):
        """Test tool parameter configuration."""
        config = {
            "tool_params": {
                "analyze_image": {"blur_threshold": 75.0}
            }
        }
        
        manager = ToolsManager(config)
        
        # Get tool and test parameter
        analyze_img = manager.get_tool("analyze_image")
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = analyze_img(test_image, "blur")
        
        assert result["blur"]["threshold"] == 75.0
    
    def test_tools_namespace(self):
        """Test tools namespace creation."""
        manager = ToolsManager()
        namespace = manager.get_tools_namespace()
        
        # robo2vlm might fail due to mocking, so just check the working ones
        assert "analyze_image" in namespace
        assert "analyze_trajectory" in namespace
        
        # Test that functions are callable
        assert callable(namespace["analyze_image"])
        assert callable(namespace["analyze_trajectory"])
    
    def test_tools_prompt_generation(self):
        """Test LLM prompt generation."""
        manager = ToolsManager()
        prompt = manager.get_tools_prompt()
        
        assert "Available Tools:" in prompt
        assert "robo2vlm" in prompt
        assert "analyze_image" in prompt
        assert "Description:" in prompt
        assert "Signature:" in prompt
        assert "Usage examples:" in prompt
    
    def test_tool_enable_disable(self):
        """Test enabling and disabling tools."""
        manager = ToolsManager()
        
        # Disable a tool
        manager.disable_tool("robo2vlm")
        tools = manager.list_tools(enabled_only=True)
        assert "robo2vlm" not in tools
        
        # Re-enable the tool
        manager.enable_tool("robo2vlm")
        tools = manager.list_tools(enabled_only=True)
        assert "robo2vlm" in tools
    
    def test_direct_tool_functions(self):
        """Test using tool implementations directly."""
        # Test analyze_image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = analyze_image(test_image, "blur")
        
        assert "blur" in result
        assert "is_blurry" in result["blur"]
        assert "laplacian_variance" in result["blur"]
        
        # Test analyze_trajectory
        test_data = np.random.randn(50, 3)
        stats = analyze_trajectory(test_data, "statistics")
        
        assert "length" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["length"] == 50
    
    def test_global_tool_registration(self):
        """Test global tool registration."""
        def global_test_tool(x):
            return x * 2
        
        register_tool(
            name="global_test",
            implementation=global_test_tool,
            description="Global test tool",
            signature="global_test(x) -> Any",
            examples=["global_test(5)"]
        )
        
        # Should be available in global manager
        from robodm.agent.tools.manager import get_global_manager
        manager = get_global_manager()
        
        tools = manager.list_tools()
        assert "global_test" in tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])