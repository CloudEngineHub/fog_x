"""
Demo of the new extensible tools system for RoboDM Agent.

The new registration-based architecture provides:
- Automatic tool registration with decorators
- Extensible number of tools
- Type-safe tool metadata
- Flexible configuration management
- Clean separation of concerns
"""

from typing import Any, Dict

import numpy as np

from robodm.agent.tools import (BaseTool, ToolMetadata, ToolsManager,
                                analyze_image, analyze_trajectory,
                                create_analysis_config, create_custom_config,
                                create_minimal_config, create_vision_config,
                                get_registry, register_tool)


def demo_clean_architecture():
    """Demonstrate the new registration-based architecture."""
    print("=== New Registration-Based Architecture Demo ===")

    # 1. Direct tool usage (legacy functions)
    print("\n--- Direct Tool Usage (Legacy API) ---")
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = analyze_image(test_image, "blur")
    if isinstance(result, dict) and "blur" in result:
        print(
            f"Direct blur analysis: {result['blur'].get('is_blurry', 'N/A')}")
    else:
        print("Direct analysis completed")

    test_trajectory = np.random.randn(100, 3)
    stats = analyze_trajectory(test_trajectory, "statistics")
    if isinstance(stats, dict) and "length" in stats:
        print(
            f"Direct trajectory stats: length={stats['length']}, mean={np.array(stats['mean'])[:2]}"
        )
    else:
        print("Direct trajectory analysis completed")

    # 2. Managed tool usage (with configuration)
    print("\n--- Managed Tool Usage (New API) ---")
    manager = ToolsManager()
    print(f"Available tools: {manager.list_tools()}")

    # Get configured tool instances
    if "analyze_image" in manager.list_tools():
        analyze_img = manager.get_tool("analyze_image")
        managed_result = analyze_img(test_image, "blur")
        if isinstance(managed_result, dict) and "blur" in managed_result:
            print(
                f"Managed blur analysis: {managed_result['blur'].get('is_blurry', 'N/A')}"
            )
        else:
            print("Managed analysis completed")

    # 3. Show tool metadata
    print("\n--- Tool Metadata ---")
    registry = get_registry()
    for tool_name in manager.list_tools()[:2]:  # Show first 2 tools
        metadata = registry.get_tool_metadata(tool_name)
        if metadata:
            print(f"{tool_name}: {metadata.description}")
            if metadata.examples:
                print(f"  Example: {metadata.examples[0]}")


def demo_configuration_system():
    """Demonstrate the configuration system."""
    print("\n=== Configuration System Demo ===")

    configs = {
        "Vision-focused":
        create_vision_config(),
        "Analysis-focused":
        create_analysis_config(),
        "Minimal":
        create_minimal_config(),
        "Custom":
        create_custom_config(
            enabled_tools=["analyze_image", "analyze_trajectory"],
            tool_parameters={
                "analyze_image": {
                    "blur_threshold": 60.0
                },
                "analyze_trajectory": {
                    "anomaly_threshold": 2.0
                },
            },
        ),
    }

    for name, config in configs.items():
        print(f"\n--- {name} Configuration ---")
        try:
            manager = ToolsManager(config=config)
            print(f"Enabled tools: {manager.list_tools()}")

            if "analyze_image" in manager.list_tools():
                # Test configuration
                analyze_img = manager.get_tool("analyze_image")
                test_image = np.ones((32, 32, 3), dtype=np.uint8) * 128
                result = analyze_img(test_image, "blur")
                if isinstance(result, dict) and "blur" in result:
                    print(
                        f"Blur threshold: {result['blur'].get('threshold', 'N/A')}"
                    )
                else:
                    print("Tool configuration successful")
        except Exception as e:
            print(f"Configuration {name} failed: {e}")
            # Fall back to default configuration
            manager = ToolsManager()
            print(f"Default tools: {manager.list_tools()}")


def demo_custom_tool_registration():
    """Demonstrate custom tool registration using the new system."""
    print("\n=== Custom Tool Registration Demo ===")

    # Example 1: Simple custom tool using decorator
    @register_tool
    class SmoothnessCalculatorTool(BaseTool):
        """Calculate trajectory smoothness using local variance."""

        def __init__(self, window_size: int = 5, **kwargs):
            super().__init__(window_size=window_size, **kwargs)
            self.window_size = window_size

        @classmethod
        def get_metadata(cls) -> ToolMetadata:
            return ToolMetadata(
                name="calculate_smoothness",
                description=
                "Calculate trajectory smoothness using local variance",
                examples=[
                    "calculate_smoothness(trajectory_data)",
                    "calculate_smoothness(trajectory_data, window_size=10)",
                ],
                tags=["trajectory", "smoothness", "analysis"],
                parameters={"window_size": 5},
            )

        def __call__(self, trajectory_data: np.ndarray) -> Dict[str, Any]:
            """Calculate trajectory smoothness."""
            if len(trajectory_data) < self.window_size:
                return {"smoothness": 0.0, "window_size": self.window_size}

            # Calculate local variance
            smoothness_scores = []
            for i in range(len(trajectory_data) - self.window_size + 1):
                window = trajectory_data[i:i + self.window_size]
                variance = np.var(window, axis=0)
                smoothness_scores.append(1.0 / (1.0 + np.mean(variance)))

            return {
                "smoothness": float(np.mean(smoothness_scores)),
                "window_size": self.window_size,
                "num_windows": len(smoothness_scores),
            }

    # Example 2: Motion classifier tool
    @register_tool
    class MotionClassifierTool(BaseTool):
        """Classify motion patterns in trajectories."""

        def __init__(
            self,
            velocity_threshold: float = 1.0,
            acceleration_threshold: float = 2.0,
            **kwargs,
        ):
            super().__init__(
                velocity_threshold=velocity_threshold,
                acceleration_threshold=acceleration_threshold,
                **kwargs,
            )
            self.velocity_threshold = velocity_threshold
            self.acceleration_threshold = acceleration_threshold

        @classmethod
        def get_metadata(cls) -> ToolMetadata:
            return ToolMetadata(
                name="classify_motion",
                description="Classify motion patterns in trajectory data",
                examples=[
                    "classify_motion(trajectory_data)",
                    "classify_motion(joint_positions)",
                ],
                tags=["motion", "classification", "trajectory"],
                parameters={
                    "velocity_threshold": 1.0,
                    "acceleration_threshold": 2.0
                },
            )

        def __call__(self, trajectory_data: np.ndarray) -> Dict[str, Any]:
            """Classify motion type."""
            if len(trajectory_data) < 3:
                return {"motion_type": "insufficient_data"}

            # Calculate velocities and accelerations
            velocities = np.diff(trajectory_data, axis=0)
            accelerations = np.diff(velocities, axis=0)

            # Calculate magnitudes
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)

            # Classify
            avg_velocity = np.mean(vel_magnitudes)
            avg_acceleration = np.mean(acc_magnitudes)

            if avg_velocity < self.velocity_threshold * 0.5:
                motion_type = "stationary"
            elif avg_acceleration < self.acceleration_threshold * 0.5:
                motion_type = "smooth"
            elif avg_acceleration > self.acceleration_threshold:
                motion_type = "jerky"
            else:
                motion_type = "normal"

            return {
                "motion_type": motion_type,
                "avg_velocity": float(avg_velocity),
                "avg_acceleration": float(avg_acceleration),
                "velocity_threshold": self.velocity_threshold,
                "acceleration_threshold": self.acceleration_threshold,
            }

    # Test the custom tools
    print("\n--- Testing Custom Tools ---")
    manager = ToolsManager()
    print(f"All available tools: {manager.list_tools()}")

    # Test smoothness calculation
    if "calculate_smoothness" in manager.list_tools():
        smoothness_tool = manager.get_tool("calculate_smoothness")
        test_smooth = np.sin(np.linspace(0, 10, 50))[:, None] * np.array(
            [1, 0.5, 0.2])
        smooth_result = smoothness_tool(test_smooth)
        print(f"Smoothness result: {smooth_result}")

    # Test motion classification
    if "classify_motion" in manager.list_tools():
        motion_tool = manager.get_tool("classify_motion")
        test_jerky = np.random.randn(50, 3) * 5  # Jerky motion
        motion_result = motion_tool(test_jerky)
        print(f"Motion classification: {motion_result}")


def demo_dynamic_configuration():
    """Demonstrate dynamic configuration management."""
    print("\n=== Dynamic Configuration Demo ===")

    # Start with minimal configuration
    manager = ToolsManager(config=create_minimal_config())
    print(f"Initial tools: {manager.list_tools()}")

    # Enable/disable tools dynamically
    print("\n--- Managing Tools ---")
    if hasattr(manager, "enable_tool"):
        manager.enable_tool("analyze_image")
        print(f"After enabling analyze_image: {manager.list_tools()}")
    else:
        print(
            "Dynamic tool enabling not available - using config-based approach"
        )
        # Create new manager with different config
        config = create_custom_config(
            enabled_tools=["robo2vlm", "analyze_image"])
        manager = ToolsManager(config=config)
        print(f"With new config: {manager.list_tools()}")

    # Test configuration updates
    print("\n--- Configuration Updates ---")
    if "analyze_image" in manager.list_tools():
        analyze_img = manager.get_tool("analyze_image")
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = analyze_img(test_image, "blur")
        if isinstance(result, dict) and "blur" in result:
            print(
                f"Current blur threshold: {result['blur'].get('threshold', 'N/A')}"
            )


def demo_llm_integration():
    """Demonstrate LLM integration features."""
    print("\n=== LLM Integration Demo ===")

    # Configuration for different scenarios
    configs = {
        "Vision tasks": create_vision_config(),
        "Analysis tasks": create_analysis_config(),
    }

    for scenario, config in configs.items():
        print(f"\n--- {scenario} ---")
        try:
            manager = ToolsManager(config=config)

            # Generate LLM prompt
            if hasattr(manager, "get_tools_prompt"):
                prompt = manager.get_tools_prompt()
                print("LLM Prompt snippet:")
                print(prompt[:300] + "..." if len(prompt) > 300 else prompt)

            # Create execution namespace
            namespace = manager.get_tools_namespace()
            print(f"Execution namespace: {list(namespace.keys())}")
        except Exception as e:
            print(f"Configuration failed: {e}")
            print("Using default configuration")
            manager = ToolsManager()
            namespace = manager.get_tools_namespace()
            print(f"Default execution namespace: {list(namespace.keys())}")


def demo_tool_metadata():
    """Demonstrate tool metadata and introspection."""
    print("\n=== Tool Metadata & Introspection Demo ===")

    manager = ToolsManager()
    registry = get_registry()

    print(f"Total registered tools: {len(manager.list_tools())}")

    for tool_name in manager.list_tools():
        print(f"\n--- {tool_name} ---")
        metadata = registry.get_tool_metadata(tool_name)
        if metadata:
            print(f"Description: {metadata.description}")
            print(f"Tags: {metadata.tags}")
            print(f"Parameters: {metadata.parameters}")
            if metadata.examples:
                print(f"Example: {metadata.examples[0]}")

        # Test tool instance
        tool_instance = manager.get_tool(tool_name)
        if tool_instance:
            print(f"Tool instance: {type(tool_instance).__name__}")


if __name__ == "__main__":
    print("RoboDM Agent - New Extensible Tools System Demo")
    print("=" * 60)

    try:
        demo_clean_architecture()
        demo_configuration_system()
        demo_custom_tool_registration()
        demo_dynamic_configuration()
        demo_llm_integration()
        demo_tool_metadata()

        print("\n" + "=" * 60)
        print("🎯 New Extensible Architecture Benefits:")
        print("✅ Automatic tool registration with decorators")
        print("✅ Type-safe tool metadata system")
        print("✅ Extensible number of tools")
        print("✅ Clean separation of concerns")
        print("✅ Flexible configuration management")
        print("✅ Easy custom tool development")
        print("✅ Backward compatibility with legacy API")
        print("✅ Unified tools manager interface")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        print(
            "This might be due to missing dependencies or configuration issues."
        )
