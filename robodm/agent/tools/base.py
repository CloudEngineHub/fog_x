"""
Base tool interface and registration system for RoboDM Agent.

This module provides the foundation for an extensible tool system where:
- Tools implement a common interface
- Tools register themselves with a global registry
- The system supports dynamic tool discovery and configuration
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities and configuration."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "robodm"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


class BaseTool(ABC):
    """
    Abstract base class for all RoboDM Agent tools.

    Tools must implement the required methods and can optionally
    override configuration and validation methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize tool with configuration parameters.

        Args:
            **kwargs: Configuration parameters for the tool
        """
        self.config = kwargs
        self.enabled = True
        self._validate_config()

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ToolMetadata:
        """
        Return metadata describing this tool.

        Returns:
            ToolMetadata instance with tool information
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute the tool's main functionality.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        pass

    def _validate_config(self):
        """
        Validate tool configuration.

        Override this method to add custom validation logic.
        Raises ValueError if configuration is invalid.
        """
        pass

    def get_signature(self) -> str:
        """
        Get the function signature for this tool.

        Returns:
            String representation of the function signature
        """
        sig = inspect.signature(self.__call__)
        params = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_str = name
            if param.annotation != inspect.Parameter.empty:
                param_str += f": {param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}"
            if param.default != inspect.Parameter.empty:
                param_str += f" = {param.default}"

            params.append(param_str)

        return_annotation = ""
        if sig.return_annotation != inspect.Signature.empty:
            return_annotation = f" -> {sig.return_annotation.__name__ if hasattr(sig.return_annotation, '__name__') else str(sig.return_annotation)}"

        return f"{self.get_metadata().name}({', '.join(params)}){return_annotation}"

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for this tool.

        Returns:
            List of usage example strings
        """
        return self.get_metadata().examples

    def enable(self):
        """Enable this tool."""
        self.enabled = True

    def disable(self):
        """Disable this tool."""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self.enabled

    def reconfigure(self, **kwargs):
        """
        Reconfigure the tool with new parameters.

        Args:
            **kwargs: New configuration parameters
        """
        self.config.update(kwargs)
        self._validate_config()


class ToolRegistry:
    """
    Global registry for managing tool registration and discovery.

    Provides a centralized system for:
    - Tool registration and discovery
    - Configuration management
    - Tool instantiation and lifecycle
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
        self._global_config: Dict[str, Any] = {}

    def register(self, tool_class: Type[BaseTool]):
        """
        Register a tool class.

        Args:
            tool_class: Tool class that inherits from BaseTool

        Raises:
            ValueError: If tool name is already registered or invalid
        """
        if not issubclass(tool_class, BaseTool):
            raise ValueError(
                f"Tool class {tool_class} must inherit from BaseTool")

        metadata = tool_class.get_metadata()

        if metadata.name in self._tool_classes:
            raise ValueError(f"Tool '{metadata.name}' is already registered")

        self._tool_classes[metadata.name] = tool_class

    def unregister(self, tool_name: str):
        """
        Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self._tool_classes:
            del self._tool_classes[tool_name]

        if tool_name in self._tool_instances:
            del self._tool_instances[tool_name]

    def get_tool(self, tool_name: str, **config) -> BaseTool:
        """
        Get a configured tool instance.

        Args:
            tool_name: Name of the tool
            **config: Configuration parameters for the tool

        Returns:
            Configured tool instance

        Raises:
            ValueError: If tool is not registered
        """
        if tool_name not in self._tool_classes:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        # Create instance key based on configuration
        config_key = str(sorted(config.items()))
        instance_key = f"{tool_name}_{hash(config_key)}"

        # Return cached instance if available
        if instance_key in self._tool_instances:
            return self._tool_instances[instance_key]

        # Merge global config with tool-specific config
        final_config = self._global_config.get(tool_name, {}).copy()
        final_config.update(config)

        # Create new instance
        tool_class = self._tool_classes[tool_name]
        tool_instance = tool_class(**final_config)

        # Cache the instance
        self._tool_instances[instance_key] = tool_instance

        return tool_instance

    def list_tools(self, enabled_only: bool = False) -> List[str]:
        """
        List registered tool names.

        Args:
            enabled_only: If True, only return enabled tools

        Returns:
            List of tool names
        """
        if not enabled_only:
            return list(self._tool_classes.keys())

        enabled_tools = []
        for tool_name in self._tool_classes.keys():
            try:
                tool = self.get_tool(tool_name)
                if tool.is_enabled():
                    enabled_tools.append(tool_name)
            except Exception:
                # Skip tools that fail to instantiate
                continue

        return enabled_tools

    def get_tool_metadata(self, tool_name: str) -> ToolMetadata:
        """
        Get metadata for a registered tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool metadata

        Raises:
            ValueError: If tool is not registered
        """
        if tool_name not in self._tool_classes:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        return self._tool_classes[tool_name].get_metadata()

    def configure_tool(self, tool_name: str, **config):
        """
        Set global configuration for a tool.

        Args:
            tool_name: Name of the tool
            **config: Configuration parameters
        """
        if tool_name not in self._global_config:
            self._global_config[tool_name] = {}

        self._global_config[tool_name].update(config)

        # Clear cached instances for this tool
        keys_to_remove = [
            key for key in self._tool_instances.keys()
            if key.startswith(f"{tool_name}_")
        ]
        for key in keys_to_remove:
            del self._tool_instances[key]

    def get_tools_namespace(self,
                            tool_names: Optional[List[str]] = None,
                            **tool_configs) -> Dict[str, BaseTool]:
        """
        Create a namespace of tool instances for code execution.

        Args:
            tool_names: List of tool names to include (None for all enabled)
            **tool_configs: Configuration for specific tools

        Returns:
            Dictionary mapping tool names to instances
        """
        if tool_names is None:
            tool_names = self.list_tools(enabled_only=True)

        namespace = {}
        for tool_name in tool_names:
            try:
                config = tool_configs.get(tool_name, {})
                tool = self.get_tool(tool_name, **config)
                if tool.is_enabled():
                    namespace[tool_name] = tool
            except Exception as e:
                # Log warning but continue with other tools
                print(f"Warning: Failed to load tool '{tool_name}': {e}")

        return namespace

    def get_tools_documentation(self,
                                tool_names: Optional[List[str]] = None) -> str:
        """
        Generate documentation for tools.

        Args:
            tool_names: List of tool names to document (None for all enabled)

        Returns:
            Formatted documentation string
        """
        if tool_names is None:
            tool_names = self.list_tools(enabled_only=True)

        if not tool_names:
            return "# No tools available"

        doc_lines = ["# Available Tools"]

        for tool_name in sorted(tool_names):
            try:
                metadata = self.get_tool_metadata(tool_name)
                tool = self.get_tool(tool_name)

                doc_lines.extend([
                    f"\n## {metadata.name}",
                    f"**Description:** {metadata.description}",
                    f"**Version:** {metadata.version}",
                    f"**Signature:** `{tool.get_signature()}`",
                ])

                if metadata.tags:
                    doc_lines.append(f"**Tags:** {', '.join(metadata.tags)}")

                examples = tool.get_usage_examples()
                if examples:
                    doc_lines.append("**Examples:**")
                    for example in examples:
                        doc_lines.append(f"```python\n{example}\n```")

            except Exception as e:
                doc_lines.append(f"\n## {tool_name} (Error: {e})")

        return "\n".join(doc_lines)

    def clear_cache(self):
        """Clear all cached tool instances."""
        self._tool_instances.clear()

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tool_classes)

    def __repr__(self) -> str:
        """String representation of registry."""
        enabled_count = len(self.list_tools(enabled_only=True))
        total_count = len(self._tool_classes)
        return f"ToolRegistry({enabled_count}/{total_count} tools enabled)"


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.

    Returns:
        The global ToolRegistry instance
    """
    return _global_registry


def register_tool(tool_class: Type[BaseTool]):
    """
    Decorator for registering tools with the global registry.

    Args:
        tool_class: Tool class to register

    Returns:
        The tool class (for use as decorator)

    Example:
        @register_tool
        class MyCustomTool(BaseTool):
            # ... implementation
    """
    _global_registry.register(tool_class)
    return tool_class
