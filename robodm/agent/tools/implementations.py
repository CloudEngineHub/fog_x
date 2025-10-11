"""
Tool implementations for RoboDM Agent using the new registration system.

This module contains concrete tool implementations that inherit from BaseTool
and register themselves with the global registry.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from .base import BaseTool, ToolMetadata, register_tool
except ImportError:
    # For backward compatibility when base module is not available
    BaseTool = object
    ToolMetadata = dict

    def register_tool(cls):
        return cls


# Handle optional dependencies gracefully
try:
    from PIL import Image
except ImportError:

    class Image:

        @staticmethod
        def fromarray(array, mode=None):
            return MockImage()


class MockImage:

    def save(self, buffer, format=None):
        buffer.write(b"mock_image_data")


from ..vlm_service import get_vlm_service


# =============================================================================
# VISION-LANGUAGE MODEL TOOL
# =============================================================================


class VisionLanguageModel:
    """Vision-language model for analyzing images using shared VLM service."""

    def __init__(self,
                 model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
                 temperature: float = 0.1,
                 max_tokens: int = 256,
                 trust_remote_code: bool = True,
                 **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs
        
        # Initialize shared VLM service
        self.vlm_service = get_vlm_service()
        self.vlm_service.initialize(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

    def __call__(self, frame: Union[np.ndarray, Image.Image, List[Union[np.ndarray, Image.Image]]],
                 prompt: str) -> str:
        """Analyze image(s) with shared VLM service.

        Accepts a single frame or a list of frames; if a list is provided,
        the service will analyze all images together with the same prompt.
        """
        if isinstance(frame, list):
            return self.vlm_service.analyze_images(frame, prompt)
        return self.vlm_service.analyze_image(frame, prompt)


# =============================================================================
# IMAGE ANALYSIS TOOLS
# =============================================================================


def analyze_image(frame: np.ndarray,
                  analysis_type: str = "all",
                  **kwargs) -> Dict[str, Any]:
    """
    Analyze image properties.

    Args:
        frame: Input image as numpy array
        analysis_type: Type of analysis ('blur', 'brightness', 'features', 'all')
        **kwargs: Additional parameters (blur_threshold, brightness_threshold)

    Returns:
        Dictionary with analysis results
    """
    blur_threshold = kwargs.get("blur_threshold", 100.0)
    brightness_threshold = kwargs.get("brightness_threshold", 0.3)

    try:
        results = {}

        if analysis_type in ["blur", "all"]:
            # Blur detection using Laplacian variance
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame

            laplacian_var = np.var(np.gradient(gray))
            results["blur"] = {
                "is_blurry": laplacian_var < blur_threshold,
                "laplacian_variance": float(laplacian_var),
                "threshold": blur_threshold,
            }

        if analysis_type in ["brightness", "all"]:
            # Brightness analysis
            mean_brightness = np.mean(frame) / 255.0
            results["brightness"] = {
                "mean_brightness":
                float(mean_brightness),
                "is_dark":
                mean_brightness < brightness_threshold,
                "is_bright":
                mean_brightness > (1.0 - brightness_threshold),
                "is_normal":
                brightness_threshold <= mean_brightness <=
                (1.0 - brightness_threshold),
            }

        if analysis_type in ["features", "all"]:
            # Basic feature extraction
            results["features"] = {
                "shape":
                list(frame.shape),
                "mean_rgb": (np.mean(frame, axis=(0, 1)).tolist() if len(
                    frame.shape) == 3 else float(np.mean(frame))),
                "std_rgb": (np.std(frame, axis=(0, 1)).tolist() if len(
                    frame.shape) == 3 else float(np.std(frame))),
                "min_val":
                float(np.min(frame)),
                "max_val":
                float(np.max(frame)),
            }

        return results

    except Exception as e:
        return {"error": f"Error in analyze_image: {str(e)}"}


# =============================================================================
# TRAJECTORY ANALYSIS TOOLS
# =============================================================================


def analyze_trajectory(data: np.ndarray,
                       analysis_type: str = "statistics",
                       **kwargs) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Analyze trajectory data.

    Args:
        data: Trajectory data as numpy array
        analysis_type: Type of analysis ('velocity', 'statistics', 'anomalies', 'smooth')
        **kwargs: Additional parameters (anomaly_threshold, min_length, smoothing_window)

    Returns:
        Analysis results (array for velocity/smooth, dict for others)
    """
    anomaly_threshold = kwargs.get("anomaly_threshold", 3.0)
    min_length = kwargs.get("min_length", 10)
    smoothing_window = kwargs.get("smoothing_window", 5)

    try:
        if analysis_type == "velocity":
            # Compute velocity (first derivative)
            return np.diff(data, axis=0)

        elif analysis_type == "statistics":
            # Compute basic statistics
            return {
                "length": len(data),
                "mean": np.mean(data, axis=0).tolist(),
                "std": np.std(data, axis=0).tolist(),
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist(),
                "is_long_enough": len(data) >= min_length,
            }

        elif analysis_type == "anomalies":
            # Detect anomalies using statistical thresholding
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)

            anomalies = np.any(np.abs(data - mean_val)
                               > anomaly_threshold * std_val,
                               axis=1)

            return {
                "anomaly_indices": np.where(anomalies)[0].tolist(),
                "anomaly_count": int(np.sum(anomalies)),
                "anomaly_ratio": float(np.mean(anomalies)),
                "threshold_used": anomaly_threshold,
            }

        elif analysis_type == "smooth":
            # Simple moving average smoothing
            if len(data) < smoothing_window:
                return data

            smoothed = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - smoothing_window // 2)
                end_idx = min(len(data), i + smoothing_window // 2 + 1)
                smoothed[i] = np.mean(data[start_idx:end_idx], axis=0)

            return smoothed

        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

    except Exception as e:
        return {"error": f"Error in analyze_trajectory: {str(e)}"}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def detect_scene_changes(images: np.ndarray,
                         vlm_func: callable,
                         threshold: float = 0.5) -> list:
    """
    Detect scene changes in a sequence of images using VLM.

    Args:
        images: Array of images with shape (T, H, W, C)
        vlm_func: Vision-language model function
        threshold: Similarity threshold for scene change detection

    Returns:
        List of frame indices where scene changes occur
    """
    if len(images) < 2:
        return []

    scene_changes = []
    prev_scene = vlm_func(images[0], "Describe the scene in one sentence.")

    for i in range(1, len(images)):
        curr_scene = vlm_func(images[i], "Describe the scene in one sentence.")

        # Simple similarity check
        similarity_prompt = f"Are these two scenes similar? Scene 1: {prev_scene}. Scene 2: {curr_scene}. Answer with yes or no."
        similarity = vlm_func(images[i], similarity_prompt).lower()

        if "no" in similarity:
            scene_changes.append(i)
            prev_scene = curr_scene

    return scene_changes


def extract_keyframes(images: np.ndarray, num_keyframes: int = 5) -> tuple:
    """
    Extract keyframes from image sequence.

    Args:
        images: Array of images with shape (T, H, W, C)
        num_keyframes: Number of keyframes to extract

    Returns:
        Tuple of (keyframe_indices, keyframes)
    """
    if len(images) <= num_keyframes:
        return list(range(len(images))), images

    # Simple uniform sampling
    indices = np.linspace(0, len(images) - 1, num_keyframes, dtype=int)
    return indices.tolist(), images[indices]


# =============================================================================
# NEW REGISTRATION-BASED TOOL IMPLEMENTATIONS
# =============================================================================


@register_tool
class VisionLanguageModelTool(BaseTool):
    """Vision-language model tool for analyzing robotic frames."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 256,
        **kwargs,
    ):
        """
        Initialize VisionLanguageModel tool.

        Args:
            model: VLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        super().__init__(model=model,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         **kwargs)

        # Initialize shared VLM service
        self.vlm_service = get_vlm_service()
        self.vlm_service.initialize(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=kwargs.get("trust_remote_code", True),
            start_command=kwargs.get("start_command"),
            **{k: v for k, v in kwargs.items() if k not in ["trust_remote_code", "start_command"]}
        )
        
        self.vlm = VisionLanguageModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=kwargs.get("trust_remote_code", True),
            **{k: v for k, v in kwargs.items() if k not in ["trust_remote_code", "start_command"]}
        )

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="robo2vlm",
            description="Vision-language model for analyzing robotic frames",
            examples=[
                'robo2vlm(frame, "Is there any object occluded or partially hidden?")',
                'robo2vlm(frame, "What type of scene is this? (kitchen, office, outdoor)")',
                'robo2vlm(frame, "How many objects are visible in this image?")',
                'robo2vlm(frame, "Describe the lighting conditions in this image")',
            ],
            tags=["vision", "language", "analysis", "robotic"],
            parameters={
                "model": "Qwen/Qwen2.5-VL-32B-Instruct",
                "temperature": 0.1,
                "max_tokens": 256
            },
        )

    def _validate_config(self):
        """Validate tool configuration."""
        if (self.config.get("temperature", 0.1) < 0
                or self.config.get("temperature", 0.1) > 2.0):
            raise ValueError("Temperature must be between 0 and 2.0")

        if self.config.get("max_tokens", 256) <= 0:
            raise ValueError("max_tokens must be positive")

    def __call__(self, frame: Union[np.ndarray, Image.Image, List[Union[np.ndarray, Image.Image]]],
                 prompt: str) -> str:
        """
        Analyze image(s) with SGLang vision-language model.

        Args:
            frame: Input image as numpy array or PIL Image, or list of images
            prompt: Natural language prompt/question about the image

        Returns:
            String response from the vision-language model
        """
        return self.vlm(frame, prompt)

    def reconfigure(self, **kwargs):
        """Reconfigure the tool with new parameters."""
        super().reconfigure(**kwargs)
        
        # Reinitialize shared VLM service with new config
        self.vlm_service.initialize(
            model=self.config.get("model", "Qwen/Qwen2.5-VL-32B-Instruct"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 256),
            trust_remote_code=self.config.get("trust_remote_code", True),
            start_command=self.config.get("start_command"),
            **{k: v for k, v in self.config.items() 
               if k not in ["model", "temperature", "max_tokens", "trust_remote_code", "start_command"]}
        )
        
        # Recreate VLM instance with new config
        self.vlm = VisionLanguageModel(
            model=self.config.get("model", "Qwen/Qwen2.5-VL-32B-Instruct"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 256),
            trust_remote_code=self.config.get("trust_remote_code", True),
            **{k: v for k, v in self.config.items() 
               if k not in ["model", "temperature", "max_tokens", "trust_remote_code", "start_command"]}
        )


@register_tool
class ImageAnalysisTool(BaseTool):
    """Tool for image analysis operations."""

    def __init__(self,
                 blur_threshold: float = 100.0,
                 brightness_threshold: float = 0.3,
                 **kwargs):
        """
        Initialize ImageAnalysisTool.

        Args:
            blur_threshold: Threshold for blur detection
            brightness_threshold: Threshold for brightness analysis
            **kwargs: Additional configuration
        """
        super().__init__(
            blur_threshold=blur_threshold,
            brightness_threshold=brightness_threshold,
            **kwargs,
        )

        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="analyze_image",
            description=
            "Analyze image properties including blur detection, brightness analysis, and feature extraction",
            examples=[
                'analyze_image(frame, "blur")',
                'analyze_image(frame, "brightness")',
                'analyze_image(frame, "features")',
                'analyze_image(frame, "all")',
            ],
            tags=["image", "analysis", "computer-vision"],
            parameters={
                "blur_threshold": 100.0,
                "brightness_threshold": 0.3
            },
        )

    def _validate_config(self):
        """Validate tool configuration."""
        if self.config.get("blur_threshold", 100.0) <= 0:
            raise ValueError("blur_threshold must be positive")

        if not 0 <= self.config.get("brightness_threshold", 0.3) <= 1:
            raise ValueError("brightness_threshold must be between 0 and 1")

    def __call__(self,
                 frame: np.ndarray,
                 analysis_type: str = "all") -> Dict[str, Any]:
        """
        Analyze image properties.

        Args:
            frame: Input image as numpy array
            analysis_type: Type of analysis ('blur', 'brightness', 'features', 'all')

        Returns:
            Dictionary with analysis results
        """
        try:
            results = {}

            if analysis_type in ["blur", "all"]:
                results["blur"] = self._detect_blur(frame)

            if analysis_type in ["brightness", "all"]:
                results["brightness"] = self._detect_brightness(frame)

            if analysis_type in ["features", "all"]:
                results["features"] = self._extract_features(frame)

            return results

        except Exception as e:
            return {"error": f"Error in analyze_image: {str(e)}"}

    def _detect_blur(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect if image is blurry using Laplacian variance."""
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame

        laplacian_var = np.var(np.gradient(gray))

        return {
            "is_blurry": laplacian_var < self.blur_threshold,
            "laplacian_variance": float(laplacian_var),
            "threshold": self.blur_threshold,
        }

    def _detect_brightness(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze brightness of image."""
        mean_brightness = np.mean(frame) / 255.0

        return {
            "mean_brightness":
            float(mean_brightness),
            "is_dark":
            mean_brightness < self.brightness_threshold,
            "is_bright":
            mean_brightness > (1.0 - self.brightness_threshold),
            "is_normal":
            self.brightness_threshold <= mean_brightness <=
            (1.0 - self.brightness_threshold),
        }

    def _extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract basic image features."""
        return {
            "shape":
            list(frame.shape),
            "mean_rgb": (np.mean(frame, axis=(0, 1)).tolist()
                         if len(frame.shape) == 3 else float(np.mean(frame))),
            "std_rgb": (np.std(frame, axis=(0, 1)).tolist()
                        if len(frame.shape) == 3 else float(np.std(frame))),
            "min_val":
            float(np.min(frame)),
            "max_val":
            float(np.max(frame)),
        }


@register_tool
class TrajectoryAnalysisTool(BaseTool):
    """Tool for trajectory-level analysis operations."""

    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        min_length: int = 10,
        smoothing_window: int = 5,
        **kwargs,
    ):
        """
        Initialize TrajectoryAnalysisTool.

        Args:
            anomaly_threshold: Threshold for anomaly detection (standard deviations)
            min_length: Minimum trajectory length threshold
            smoothing_window: Window size for smoothing operations
            **kwargs: Additional configuration
        """
        super().__init__(
            anomaly_threshold=anomaly_threshold,
            min_length=min_length,
            smoothing_window=smoothing_window,
            **kwargs,
        )

        self.anomaly_threshold = anomaly_threshold
        self.min_length = min_length
        self.smoothing_window = smoothing_window

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="analyze_trajectory",
            description=
            "Analyze trajectory data including velocity computation, statistics, anomaly detection, and smoothing",
            examples=[
                'analyze_trajectory(trajectory["joint_positions"], "velocity")',
                'analyze_trajectory(trajectory["actions"], "statistics")',
                'analyze_trajectory(trajectory["sensor_data"], "anomalies")',
                'analyze_trajectory(trajectory["noisy_data"], "smooth")',
            ],
            tags=["trajectory", "analysis", "robotics"],
            parameters={
                "anomaly_threshold": 3.0,
                "min_length": 10,
                "smoothing_window": 5,
            },
        )

    def _validate_config(self):
        """Validate tool configuration."""
        if self.config.get("anomaly_threshold", 3.0) <= 0:
            raise ValueError("anomaly_threshold must be positive")

        if self.config.get("min_length", 10) <= 0:
            raise ValueError("min_length must be positive")

        if self.config.get("smoothing_window", 5) <= 0:
            raise ValueError("smoothing_window must be positive")

    def __call__(
        self,
        data: np.ndarray,
        analysis_type: str = "statistics"
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Perform trajectory analysis operation.

        Args:
            data: Trajectory data as numpy array
            analysis_type: Type of analysis ('velocity', 'statistics', 'anomalies', 'smooth')

        Returns:
            Analysis results (array for velocity/smooth, dict for others)
        """
        try:
            if analysis_type == "velocity":
                return self._compute_velocity(data)
            elif analysis_type == "statistics":
                return self._compute_statistics(data)
            elif analysis_type == "anomalies":
                return self._detect_anomalies(data)
            elif analysis_type == "smooth":
                return self._smooth_trajectory(data)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

        except Exception as e:
            return {"error": f"Error in analyze_trajectory: {str(e)}"}

    def _compute_velocity(self, data: np.ndarray) -> np.ndarray:
        """Compute velocity from position data."""
        return np.diff(data, axis=0)

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics for trajectory data."""
        return {
            "length": len(data),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "is_long_enough": len(data) >= self.min_length,
        }

    def _detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in trajectory data."""
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)

        anomalies = np.any(np.abs(data - mean_val)
                           > self.anomaly_threshold * std_val,
                           axis=1)

        return {
            "anomaly_indices": np.where(anomalies)[0].tolist(),
            "anomaly_count": int(np.sum(anomalies)),
            "anomaly_ratio": float(np.mean(anomalies)),
            "threshold_used": self.anomaly_threshold,
        }

    def _smooth_trajectory(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to trajectory data."""
        if len(data) < self.smoothing_window:
            return data

        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(data), i + self.smoothing_window // 2 + 1)
            smoothed[i] = np.mean(data[start_idx:end_idx], axis=0)

        return smoothed
