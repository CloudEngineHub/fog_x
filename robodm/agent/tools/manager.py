"""
Tool manager for RoboDM Agent - coordinates tool registration and lifecycle.

This module provides a high-level interface for managing tools, built on top
of the base registration system. It handles configuration, tool discovery,
and provides a clean API for the Agent class.
"""

from typing import Any, Dict, List, Optional, Type

from .base import BaseTool, ToolRegistry, get_registry


class ToolsManager:
    """
    High-level tool management interface for RoboDM Agent.

    Provides configuration management, tool discovery, and execution
    context creation for the Agent system.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ToolsManager.

        Args:
            registry: Tool registry to use (uses global if None)
            config: Initial configuration dictionary
        """
        self.registry = registry or get_registry()
        self.config = config or {}

        # Apply initial configuration
        self._apply_config()

    def _apply_config(self):
        """Apply configuration to registry and tools."""
        # Configure individual tools
        tool_configs = self.config.get("tools", {})
        for tool_name, tool_config in tool_configs.items():
            if isinstance(tool_config, dict):
                self.registry.configure_tool(tool_name, **tool_config)

        # Handle disabled tools
        disabled_tools = self.config.get("disabled_tools", [])
        for tool_name in disabled_tools:
            try:
                tool = self.registry.get_tool(tool_name)
                tool.disable()
            except ValueError:
                # Tool not registered, skip
                pass

    def register_tool(self, tool_class: Type[BaseTool]):
        """
        Register a new tool class.

        Args:
            tool_class: Tool class inheriting from BaseTool
        """
        self.registry.register(tool_class)

    def unregister_tool(self, tool_name: str):
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to unregister
        """
        self.registry.unregister(tool_name)

    def get_tool(self, tool_name: str, **config) -> BaseTool:
        """
        Get a configured tool instance.

        Args:
            tool_name: Name of the tool
            **config: Additional configuration parameters

        Returns:
            Configured tool instance
        """
        return self.registry.get_tool(tool_name, **config)

    def list_tools(self, enabled_only: bool = True) -> List[str]:
        """
        List available tools.

        Args:
            enabled_only: Only return enabled tools

        Returns:
            List of tool names
        """
        return self.registry.list_tools(enabled_only=enabled_only)

    def enable_tool(self, tool_name: str):
        """
        Enable a tool.

        Args:
            tool_name: Name of tool to enable
        """
        try:
            tool = self.registry.get_tool(tool_name)
            tool.enable()

            # Update config
            disabled_tools = self.config.get("disabled_tools", [])
            if tool_name in disabled_tools:
                disabled_tools.remove(tool_name)

        except ValueError as e:
            raise ValueError(f"Cannot enable tool '{tool_name}': {e}")

    def disable_tool(self, tool_name: str):
        """
        Disable a tool.

        Args:
            tool_name: Name of tool to disable
        """
        try:
            tool = self.registry.get_tool(tool_name)
            tool.disable()

            # Update config
            if "disabled_tools" not in self.config:
                self.config["disabled_tools"] = []
            if tool_name not in self.config["disabled_tools"]:
                self.config["disabled_tools"].append(tool_name)

        except ValueError as e:
            raise ValueError(f"Cannot disable tool '{tool_name}': {e}")

    def configure_tool(self, tool_name: str, **config):
        """
        Configure a tool with new parameters.

        Args:
            tool_name: Name of tool to configure
            **config: Configuration parameters
        """
        # Update manager config
        if "tools" not in self.config:
            self.config["tools"] = {}
        if tool_name not in self.config["tools"]:
            self.config["tools"][tool_name] = {}

        self.config["tools"][tool_name].update(config)

        # Apply to registry
        self.registry.configure_tool(tool_name, **config)

    def get_tools_namespace(
            self,
            tool_names: Optional[List[str]] = None) -> Dict[str, BaseTool]:
        """
        Create namespace of tools for code execution.

        Args:
            tool_names: Specific tools to include (None for all enabled)

        Returns:
            Dictionary mapping tool names to instances
        """
        tool_configs = self.config.get("tools", {})
        return self.registry.get_tools_namespace(tool_names, **tool_configs)

    def get_tools_prompt(self) -> str:
        """
        Get tools documentation for LLM prompts.

        Returns:
            Formatted tools documentation
        """
        enabled_tools = self.list_tools(enabled_only=True)
        return self.registry.get_tools_documentation(enabled_tools)

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool information
        """
        try:
            metadata = self.registry.get_tool_metadata(tool_name)
            tool = self.registry.get_tool(tool_name)

            return {
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "author": metadata.author,
                "tags": metadata.tags,
                "signature": tool.get_signature(),
                "examples": tool.get_usage_examples(),
                "enabled": tool.is_enabled(),
                "config": tool.config,
            }
        except ValueError as e:
            raise ValueError(f"Tool '{tool_name}' not found: {e}")

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update manager configuration.

        Args:
            new_config: New configuration to merge
        """
        self.config.update(new_config)
        self._apply_config()

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Copy of current configuration
        """
        return self.config.copy()

    def clear_cache(self):
        """Clear tool instance cache."""
        self.registry.clear_cache()

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool registry.

        Returns:
            Dictionary with registry statistics
        """
        all_tools = self.registry.list_tools(enabled_only=False)
        enabled_tools = self.registry.list_tools(enabled_only=True)

        return {
            "total_tools": len(all_tools),
            "enabled_tools": len(enabled_tools),
            "disabled_tools": len(all_tools) - len(enabled_tools),
            "cached_instances": len(self.registry._tool_instances),
            "tools": all_tools,
        }

    def __repr__(self) -> str:
        """String representation of ToolsManager."""
        stats = self.get_registry_stats()
        return f"ToolsManager({stats['enabled_tools']}/{stats['total_tools']} tools enabled)"


# Legacy compatibility - will be removed in future versions
class LegacyToolsManager(ToolsManager):
    """Legacy compatibility wrapper for the old ToolsManager interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with legacy configuration format."""
        super().__init__(config=config)

        # Import and register legacy tools for backward compatibility
        self._register_legacy_tools()

    def _register_legacy_tools(self):
        """Register legacy tools for backward compatibility."""
        try:
            from .base import ToolMetadata, register_tool
            from .implementations import (VisionLanguageModel, analyze_image,
                                          analyze_trajectory)

            # Register VisionLanguageModel
            @register_tool
            class VisionLanguageModelTool(VisionLanguageModel):

                @classmethod
                def get_metadata(cls) -> ToolMetadata:
                    return ToolMetadata(
                        name="robo2vlm",
                        description=
                        "Vision-language model for analyzing robotic frames",
                        examples=[
                            'robo2vlm(frame, "Is there any object occluded or partially hidden?")',
                            'robo2vlm(frame, "What type of scene is this? (kitchen, office, outdoor)")',
                            'robo2vlm(frame, "How many objects are visible in this image?")',
                        ],
                        tags=["vision", "language", "analysis"],
                    )

            # Register function-based tools
            class FunctionBaseTool(BaseTool):

                def __init__(self, func, metadata, **kwargs):
                    super().__init__(**kwargs)
                    self.func = func
                    self.metadata = metadata

                @classmethod
                def get_metadata(cls) -> ToolMetadata:
                    return cls.metadata

                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

            @register_tool
            class AnalyzeImageTool(FunctionBaseTool):

                def __init__(self, **kwargs):
                    metadata = ToolMetadata(
                        name="analyze_image",
                        description=
                        "Analyze image properties (blur, brightness, features)",
                        examples=[
                            'analyze_image(frame, "blur")',
                            'analyze_image(frame, "brightness")',
                            'analyze_image(frame, "all")',
                        ],
                        tags=["image", "analysis"],
                    )
                    super().__init__(analyze_image, metadata, **kwargs)

            @register_tool
            class AnalyzeTrajectoryTool(FunctionBaseTool):

                def __init__(self, **kwargs):
                    metadata = ToolMetadata(
                        name="analyze_trajectory",
                        description=
                        "Analyze trajectory data (velocity, statistics, anomalies)",
                        examples=[
                            'analyze_trajectory(trajectory["joint_positions"], "velocity")',
                            'analyze_trajectory(trajectory["actions"], "statistics")',
                            'analyze_trajectory(trajectory["sensor_data"], "anomalies")',
                        ],
                        tags=["trajectory", "analysis"],
                    )
                    super().__init__(analyze_trajectory, metadata, **kwargs)

        except ImportError:
            # Legacy tools not available
            pass
