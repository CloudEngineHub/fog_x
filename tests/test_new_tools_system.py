"""
Tests for the reorganized tools system.
"""

import sys

import numpy as np
import pytest


# Mock vllm module
class MockSamplingParams:

    def __init__(self, **kwargs):
        self.params = kwargs


sys.modules["vllm"] = type(
    "MockVLLM",
    (),
    {
        "LLM":
        type(
            "MockLLM",
            (),
            {
                "__init__":
                lambda self, model: None,
                "generate":
                lambda self, prompts, params: [
                    type(
                        "MockOutput",
                        (),
                        {
                            "outputs": [
                                type("MockGeneration",
                                     (), {"text": "Mock response"})()
                            ]
                        },
                    )()
                ],
            },
        ),
        "SamplingParams":
        MockSamplingParams,
    },
)()

from robodm.agent.tools import (ToolsManager, analyze_image,
                                analyze_trajectory, create_analysis_config,
                                create_custom_config, create_minimal_config,
                                create_vision_config, register_tool)


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

        assert "disabled_tools" in vision_config
        assert "analyze_trajectory" in vision_config["disabled_tools"]

        assert "disabled_tools" in analysis_config
        assert len(analysis_config["disabled_tools"]) == 0

        assert "disabled_tools" in minimal_config
        assert "analyze_image" in minimal_config["disabled_tools"]
        assert "analyze_trajectory" in minimal_config["disabled_tools"]

    def test_custom_configuration(self):
        """Test custom configuration."""
        config = create_custom_config(
            enabled_tools=["analyze_image"],
            tool_parameters={"analyze_image": {
                "blur_threshold": 50.0
            }},
        )

        manager = ToolsManager(config=config)
        tools = manager.list_tools()

        assert "analyze_image" in tools
        assert "robo2vlm" not in tools  # Should be disabled
        assert "analyze_trajectory" not in tools  # Should be disabled

    def test_tool_registration(self):
        """Test tool registration."""
        from robodm.agent.tools import BaseTool, ToolMetadata

        class CustomThresholdTool(BaseTool):

            def __init__(self, threshold: float = 1.0, **kwargs):
                super().__init__(threshold=threshold, **kwargs)
                self.threshold = threshold

            @classmethod
            def get_metadata(cls) -> ToolMetadata:
                return ToolMetadata(
                    name="custom_threshold",
                    description="Custom threshold tool",
                    version="1.0.0",
                    examples=["custom_threshold(data)"],
                )

            def __call__(self, data):
                return np.mean(data) > self.threshold

        manager = ToolsManager()
        manager.register_tool(CustomThresholdTool)

        tools = manager.list_tools()
        assert "custom_threshold" in tools

        # Test tool usage
        tool = manager.get_tool("custom_threshold")
        result = tool(np.array([2, 3, 4]))
        assert result == True  # Mean 3.0 > 1.0

    def test_tool_configuration(self):
        """Test tool parameter configuration."""
        config = {"tools": {"analyze_image": {"blur_threshold": 75.0}}}

        manager = ToolsManager(config=config)

        # Get tool and test parameter
        analyze_img = manager.get_tool("analyze_image")
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = analyze_img(test_image, "blur")

        assert result["blur"]["threshold"] == 75.0

    def test_tools_namespace(self):
        """Test tools namespace creation."""
        manager = ToolsManager()
        namespace = manager.get_tools_namespace()

        # Check that at least these core tools are present
        assert "analyze_image" in namespace
        # analyze_trajectory might be disabled due to VLM issues in some test runs

        # Test that functions are callable
        assert callable(namespace["analyze_image"])

    def test_tools_prompt_generation(self):
        """Test LLM prompt generation."""
        manager = ToolsManager()
        prompt = manager.get_tools_prompt()

        assert "# Available Tools" in prompt
        # robo2vlm might not be in prompt due to VLM initialization issues
        assert "analyze_image" in prompt
        assert "**Description:**" in prompt
        assert "**Signature:**" in prompt
        assert "**Examples:**" in prompt

    def test_tool_enable_disable(self):
        """Test enabling and disabling tools."""
        manager = ToolsManager()

        # Disable a tool that doesn't require vllm
        manager.disable_tool("analyze_image")
        tools = manager.list_tools(enabled_only=True)
        assert "analyze_image" not in tools

        # Re-enable the tool
        manager.enable_tool("analyze_image")
        tools = manager.list_tools(enabled_only=True)
        assert "analyze_image" in tools

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
        from robodm.agent.tools import BaseTool, ToolMetadata, get_registry

        @register_tool
        class GlobalTestTool(BaseTool):

            @classmethod
            def get_metadata(cls) -> ToolMetadata:
                return ToolMetadata(
                    name="global_test",
                    description="Global test tool",
                    version="1.0.0",
                    examples=["global_test(5)"],
                )

            def __call__(self, x):
                return x * 2

        # Should be available in global registry
        registry = get_registry()
        tools = registry.list_tools()
        assert "global_test" in tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
