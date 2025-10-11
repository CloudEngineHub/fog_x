"""
RoboDM Agent Tools System

An extensible tools system with registration-based architecture:
- base.py: Abstract base classes and registry system
- implementations.py: Concrete tool implementations
- manager.py: High-level tool management interface
- config.py: Configuration templates and helpers

The new system provides:
- Clean separation between tool interface and implementation
- Automatic tool registration with decorators
- Flexible configuration management
- Type-safe tool metadata
- Extensible plugin architecture
"""

# Core system components
from .base import (BaseTool, ToolMetadata, ToolRegistry, get_registry,
                   register_tool)
from .config import (create_analysis_config, create_custom_config,
                     create_minimal_config, create_vision_config,
                     get_default_config, get_preset_config,
                     list_preset_configs, merge_configs, validate_config)
# Tool implementations (these auto-register when imported)
from .implementations import (  # Legacy function wrappers for backward compatibility
    ImageAnalysisTool, TrajectoryAnalysisTool, VisionLanguageModel,
    VisionLanguageModelTool, analyze_image, analyze_trajectory,
    detect_scene_changes, extract_keyframes)
from .manager import ToolsManager

__all__ = [
    # Core system
    "BaseTool",
    "ToolMetadata",
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "ToolsManager",
    # Configuration
    "create_vision_config",
    "create_analysis_config",
    "create_minimal_config",
    "create_custom_config",
    "get_preset_config",
    "list_preset_configs",
    "validate_config",
    "merge_configs",
    "get_default_config",
    # Tool implementations
    "VisionLanguageModelTool",
    "ImageAnalysisTool",
    "TrajectoryAnalysisTool",
    # Legacy compatibility
    "VisionLanguageModel",
    "analyze_image",
    "analyze_trajectory",
    "detect_scene_changes",
    "extract_keyframes",
]


# Initialize registry with default tools
def _initialize_default_tools():
    """Initialize the registry with default tools."""
    # Tools are automatically registered via decorators when imported
    # This function exists for any future initialization needs
    pass


_initialize_default_tools()


# Convenience functions for common operations
def create_manager(config_preset: str = "default",
                   **preset_kwargs) -> ToolsManager:
    """
    Create a ToolsManager with a preset configuration.

    Args:
        config_preset: Name of preset configuration to use
        **preset_kwargs: Additional arguments for preset configuration

    Returns:
        Configured ToolsManager instance

    Example:
        >>> manager = create_manager("vision", temperature=0.05)
        >>> manager = create_manager("minimal", model="llama-7b")
    """
    config = get_preset_config(config_preset, **preset_kwargs)
    return ToolsManager(config=config)


def list_available_tools() -> list:
    """
    List all available tools in the registry.

    Returns:
        List of tool names
    """
    registry = get_registry()
    return registry.list_tools(enabled_only=False)


def get_tool_documentation() -> str:
    """
    Get documentation for all available tools.

    Returns:
        Formatted documentation string
    """
    registry = get_registry()
    return registry.get_tools_documentation()


# Add convenience functions to __all__
__all__.extend(
    ["create_manager", "list_available_tools", "get_tool_documentation"])
