"""
Configuration templates and helpers for RoboDM Agent tools.

This module provides pre-defined configurations for common use cases
and helper functions for creating custom configurations.
"""

from typing import Any, Callable, Dict, List, Optional


def create_vision_config(model: str = "Llama 3.2-Vision",
                         temperature: float = 0.05,
                         max_tokens: int = 512) -> Dict[str, Any]:
    """
    Create configuration optimized for vision tasks.

    Args:
        model: VLM model name
        temperature: Lower temperature for more deterministic responses
        max_tokens: Maximum tokens for longer descriptions

    Returns:
        Configuration dictionary optimized for vision tasks
    """
    return {
        "tools": {
            "robo2vlm": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "analyze_image": {
                "blur_threshold": 80.0,  # More sensitive blur detection
                "brightness_threshold": 0.25,
            },
        },
        "disabled_tools": ["analyze_trajectory"],  # Focus on vision tasks
    }


def create_analysis_config(
    anomaly_sensitivity: float = 2.5,
    min_trajectory_length: int = 20,
    smoothing_window: int = 7,
) -> Dict[str, Any]:
    """
    Create configuration optimized for trajectory analysis.

    Args:
        anomaly_sensitivity: Lower threshold for more sensitive anomaly detection
        min_trajectory_length: Minimum length for valid trajectories
        smoothing_window: Window size for trajectory smoothing

    Returns:
        Configuration dictionary optimized for analysis tasks
    """
    return {
        "tools": {
            "analyze_trajectory": {
                "anomaly_threshold": anomaly_sensitivity,
                "min_length": min_trajectory_length,
                "smoothing_window": smoothing_window,
            },
            "analyze_image": {
                "blur_threshold": 100.0,
                "brightness_threshold": 0.3
            },
        },
        "disabled_tools": [],  # Keep all tools enabled
    }


def create_minimal_config(model: str = "Llama 3.2-Vision") -> Dict[str, Any]:
    """
    Create minimal configuration with only essential tools.

    Args:
        model: VLM model name

    Returns:
        Minimal configuration with only vision-language model
    """
    return {
        "tools": {
            "robo2vlm": {
                "model": model,
                "temperature": 0.1,
                "max_tokens": 128,  # Shorter responses for efficiency
            }
        },
        "disabled_tools": ["analyze_image", "analyze_trajectory"],
    }


def create_custom_config(
    enabled_tools: Optional[List[str]] = None,
    tool_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    disabled_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create custom configuration with specified tools and parameters.

    Args:
        enabled_tools: List of tools to enable (None = all enabled)
        tool_parameters: Parameters for specific tools
        disabled_tools: List of tools to disable

    Returns:
        Custom configuration dictionary
    """
    config: Dict[str, Any] = {}

    if tool_parameters:
        config["tools"] = tool_parameters

    if disabled_tools:
        config["disabled_tools"] = disabled_tools
    elif enabled_tools is not None:
        # If enabled_tools is specified, disable all others
        all_tools = ["robo2vlm", "analyze_image", "analyze_trajectory"]
        config["disabled_tools"] = [
            tool for tool in all_tools if tool not in enabled_tools
        ]

    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary and return list of issues.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check structure
    if not isinstance(config, dict):
        issues.append("Configuration must be a dictionary")
        return issues

    # Validate tools section
    tools_config = config.get("tools", {})
    if not isinstance(tools_config, dict):
        issues.append("'tools' section must be a dictionary")
    else:
        for tool_name, tool_config in tools_config.items():
            if not isinstance(tool_config, dict):
                issues.append(
                    f"Configuration for tool '{tool_name}' must be a dictionary"
                )
                continue

            # Validate specific tool parameters
            if tool_name == "robo2vlm":
                temp = tool_config.get("temperature", 0.1)
                if not isinstance(temp,
                                  (int, float)) or temp < 0 or temp > 2.0:
                    issues.append(
                        f"robo2vlm temperature must be between 0 and 2.0, got {temp}"
                    )

                max_tokens = tool_config.get("max_tokens", 256)
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    issues.append(
                        f"robo2vlm max_tokens must be positive integer, got {max_tokens}"
                    )

            elif tool_name == "analyze_image":
                blur_thresh = tool_config.get("blur_threshold", 100.0)
                if not isinstance(blur_thresh,
                                  (int, float)) or blur_thresh <= 0:
                    issues.append(
                        f"analyze_image blur_threshold must be positive, got {blur_thresh}"
                    )

                bright_thresh = tool_config.get("brightness_threshold", 0.3)
                if (not isinstance(bright_thresh, (int, float))
                        or not 0 <= bright_thresh <= 1):
                    issues.append(
                        f"analyze_image brightness_threshold must be between 0 and 1, got {bright_thresh}"
                    )

            elif tool_name == "analyze_trajectory":
                anom_thresh = tool_config.get("anomaly_threshold", 3.0)
                if not isinstance(anom_thresh,
                                  (int, float)) or anom_thresh <= 0:
                    issues.append(
                        f"analyze_trajectory anomaly_threshold must be positive, got {anom_thresh}"
                    )

                min_len = tool_config.get("min_length", 10)
                if not isinstance(min_len, int) or min_len <= 0:
                    issues.append(
                        f"analyze_trajectory min_length must be positive integer, got {min_len}"
                    )

                smooth_win = tool_config.get("smoothing_window", 5)
                if not isinstance(smooth_win, int) or smooth_win <= 0:
                    issues.append(
                        f"analyze_trajectory smoothing_window must be positive integer, got {smooth_win}"
                    )

    # Validate disabled_tools section
    disabled_tools = config.get("disabled_tools", [])
    if not isinstance(disabled_tools, list):
        issues.append("'disabled_tools' must be a list")
    else:
        valid_tools = ["robo2vlm", "analyze_image", "analyze_trajectory"]
        for tool in disabled_tools:
            if not isinstance(tool, str):
                issues.append(
                    f"Disabled tool name must be string, got {type(tool)}")
            elif tool not in valid_tools:
                issues.append(f"Unknown tool '{tool}' in disabled_tools")

    return issues


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configurations override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result: Dict[str, Any] = {}

    for config in configs:
        if not isinstance(config, dict):
            continue

        # Merge tools section
        if "tools" in config:
            if "tools" not in result:
                result["tools"] = {}

            for tool_name, tool_config in config["tools"].items():
                if tool_name not in result["tools"]:
                    result["tools"][tool_name] = {}
                result["tools"][tool_name].update(tool_config)

        # Override disabled_tools
        if "disabled_tools" in config:
            result["disabled_tools"] = config["disabled_tools"].copy()

        # Merge any other top-level keys
        for key, value in config.items():
            if key not in ["tools", "disabled_tools"]:
                result[key] = value

    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for all tools.

    Returns:
        Default configuration dictionary
    """
    return {
        "tools": {
            "robo2vlm": {
                "model": "Llama 3.2-Vision",
                "temperature": 0.1,
                "max_tokens": 256
            },
            "analyze_image": {
                "blur_threshold": 100.0,
                "brightness_threshold": 0.3
            },
            "analyze_trajectory": {
                "anomaly_threshold": 3.0,
                "min_length": 10,
                "smoothing_window": 5,
            },
        },
        "disabled_tools": [],
    }


# Configuration presets for common scenarios
PRESET_CONFIGS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "vision": create_vision_config,
    "analysis": create_analysis_config,
    "minimal": create_minimal_config,
    "default": get_default_config,
}


def get_preset_config(preset_name: str, **kwargs) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of the preset configuration
        **kwargs: Additional arguments to pass to the preset function

    Returns:
        Preset configuration dictionary

    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}")

    preset_func = PRESET_CONFIGS[preset_name]

    # Handle functions that don't take arguments
    if preset_name == "default":
        return preset_func()
    else:
        return preset_func(**kwargs)


def list_preset_configs() -> List[str]:
    """
    List available preset configuration names.

    Returns:
        List of preset configuration names
    """
    return list(PRESET_CONFIGS.keys())
