"""
Demo script using Llama 3.2-Vision model to classify DROID trajectories as successful or failed.

This script:
1. Downloads sample DROID trajectories (both success and failure)
2. Converts them to RoboDM format
3. Uses the Llama 3.2-Vision model to analyze trajectories
4. Demonstrates how to detect success/failure patterns
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from download_droid import DROIDDownloader
from droid_to_robodm import DROIDToRoboDMConverter

import robodm


class DROIDSuccessDetector:
    """Detect success/failure in DROID trajectories using Llama 3.2-Vision."""

    def __init__(self):
        # Initialize Llama 3.2-Vision model directly
        print("Loading Llama 3.2-Vision model...")
        self.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        # Load model and processor
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")

    def analyze_frame_with_llama_vision(self, image: np.ndarray, prompt: str) -> str:
        """
        Analyze a single frame using Llama 3.2-Vision.
        
        Args:
            image: Frame as numpy array (H, W, C)
            prompt: Text prompt for analysis
            
        Returns:
            Model response
        """
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # Create conversation format for Llama 3.2-Vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=[pil_image], 
                text=text, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1
                )
            
            # Decode response (skip the input tokens)
            generated_ids = output[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Response: {response.strip()}")
            return response.strip()
            
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return "Error"

    def analyze_trajectory_frames(self,
                                  trajectory_path: str,
                                  sample_rate: int = 10) -> Dict:
        """
        Analyze frames from a trajectory using Llama 3.2-Vision.

        Args:
            trajectory_path: Path to RoboDM trajectory file
            sample_rate: Sample every Nth frame

        Returns:
            Analysis results
        """
        # Load trajectory
        traj = robodm.Trajectory(path=trajectory_path, mode="r")
        data = traj.load()

        # Get available camera views
        camera_keys = [k for k in data.keys() if "observation/images/" in k]

        results = {
            "trajectory_path": trajectory_path,
            "frame_analyses": [],
            "overall_assessment": None,
        }

        if not camera_keys:
            print(f"No camera data found in {trajectory_path}")
            return results

        # Use the first available camera (e.g., cam_high)
        primary_camera = camera_keys[0]
        frames = data[primary_camera]

        print(f"\nAnalyzing {len(frames)} frames from {primary_camera}")

        # Sample frames for analysis
        frame_indices = range(0, len(frames), sample_rate)

        for idx in frame_indices:
            frame = frames[idx]

            # Analyze frame for task completion indicators
            prompts = [
                "Is the robot gripper holding any object? Answer yes or no.",
                "Describe what task the robot appears to be performing.",
                "Are there any signs of failure (dropped objects, collision, stuck position)?",
                "Is the task completed successfully in this frame?",
            ]

            frame_analysis = {"frame_idx": idx, "analyses": {}}

            for prompt in prompts:
                response = self.analyze_frame_with_llama_vision(frame, prompt)
                frame_analysis["analyses"][prompt] = response

            results["frame_analyses"].append(frame_analysis)

        # Analyze trajectory progression
        results["overall_assessment"] = self._assess_trajectory_success(
            results["frame_analyses"])

        traj.close()
        return results

    def _assess_trajectory_success(self, frame_analyses: List[Dict]) -> Dict:
        """
        Assess overall trajectory success based on frame analyses.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Overall assessment
        """
        # Count success/failure indicators
        success_indicators = 0
        failure_indicators = 0
        task_descriptions = []

        for analysis in frame_analyses:
            responses = analysis["analyses"]

            # Check for holding objects
            if ("yes" in responses.get(
                    "Is the robot gripper holding any object? Answer yes or no.",
                    "").lower()):
                success_indicators += 1

            # Check for failure signs
            failure_response = responses.get(
                "Are there any signs of failure (dropped objects, collision, stuck position)?",
                "",
            )
            if any(word in failure_response.lower()
                   for word in ["yes", "dropped", "collision", "stuck"]):
                failure_indicators += 1

            # Check for task completion
            if ("yes" in responses.get(
                    "Is the task completed successfully in this frame?",
                    "").lower()):
                success_indicators += 1

            # Collect task descriptions
            task_desc = responses.get(
                "Describe what task the robot appears to be performing.", "")
            if task_desc and task_desc != "Error":
                task_descriptions.append(task_desc)

        # Determine overall success
        total_frames = len(frame_analyses)
        success_rate = (success_indicators /
                        (total_frames * 2) if total_frames > 0 else 0
                        )  # *2 for two success questions
        failure_rate = failure_indicators / total_frames if total_frames > 0 else 0

        is_successful = success_rate > 0.3 and failure_rate < 0.3

        return {
            "is_successful":
            is_successful,
            "success_rate":
            success_rate,
            "failure_rate":
            failure_rate,
            "success_indicators":
            success_indicators,
            "failure_indicators":
            failure_indicators,
            "common_task":
            (max(set(task_descriptions), key=task_descriptions.count)
             if task_descriptions else "Unknown"),
        }

    def compare_trajectories(self, success_paths: List[str],
                             failure_paths: List[str]):
        """
        Compare successful and failed trajectories.

        Args:
            success_paths: List of successful trajectory paths
            failure_paths: List of failed trajectory paths
        """
        print("\n" + "=" * 60)
        print("TRAJECTORY ANALYSIS RESULTS")
        print("=" * 60)

        # Analyze successful trajectories
        print("\n--- SUCCESSFUL TRAJECTORIES ---")
        success_results = []
        for path in success_paths:
            if os.path.exists(path):
                print(f"\nAnalyzing: {os.path.basename(path)}")
                result = self.analyze_trajectory_frames(path, sample_rate=20)
                success_results.append(result)

                assessment = result["overall_assessment"]
                print(
                    f"  Predicted: {'SUCCESS' if assessment['is_successful'] else 'FAILURE'}"
                )
                print(f"  Success rate: {assessment['success_rate']:.2%}")
                print(f"  Failure rate: {assessment['failure_rate']:.2%}")
                print(f"  Task: {assessment['common_task']}")

        # Analyze failed trajectories
        print("\n--- FAILED TRAJECTORIES ---")
        failure_results = []
        for path in failure_paths:
            if os.path.exists(path):
                print(f"\nAnalyzing: {os.path.basename(path)}")
                result = self.analyze_trajectory_frames(path, sample_rate=20)
                failure_results.append(result)

                assessment = result["overall_assessment"]
                print(
                    f"  Predicted: {'SUCCESS' if assessment['is_successful'] else 'FAILURE'}"
                )
                print(f"  Success rate: {assessment['success_rate']:.2%}")
                print(f"  Failure rate: {assessment['failure_rate']:.2%}")
                print(f"  Task: {assessment['common_task']}")

        # Calculate accuracy
        print("\n--- CLASSIFICATION ACCURACY ---")
        correct_success = sum(1 for r in success_results
                              if r["overall_assessment"]["is_successful"])
        correct_failure = sum(1 for r in failure_results
                              if not r["overall_assessment"]["is_successful"])
        total_success = len(success_results)
        total_failure = len(failure_results)

        if total_success > 0:
            success_accuracy = correct_success / total_success
            print(
                f"Success detection accuracy: {success_accuracy:.2%} ({correct_success}/{total_success})"
            )

        if total_failure > 0:
            failure_accuracy = correct_failure / total_failure
            print(
                f"Failure detection accuracy: {failure_accuracy:.2%} ({correct_failure}/{total_failure})"
            )

        if total_success + total_failure > 0:
            overall_accuracy = (correct_success + correct_failure) / (
                total_success + total_failure)
            print(f"Overall accuracy: {overall_accuracy:.2%}")


def main():
    """Main demo function."""
    print("DROID Trajectory Success/Failure Detection Demo")
    print("=" * 60)

    # Step 1: Download DROID trajectories
    print("\n1. Downloading DROID trajectories...")
    downloader = DROIDDownloader()
    droid_data_dir = "./droid_data"

    if not os.path.exists(droid_data_dir):
        success_paths, failure_paths = downloader.download_sample_trajectories(
            output_dir=droid_data_dir, num_success=2, num_failure=2)
    else:
        print(f"Using existing data in {droid_data_dir}")

    # Step 2: Convert to RoboDM format
    print("\n2. Converting to RoboDM format...")
    converter = DROIDToRoboDMConverter()
    robodm_dir = "./robodm_trajectories"

    if not os.path.exists(robodm_dir):
        converter.convert_directory(droid_data_dir, robodm_dir)
    else:
        print(f"Using existing RoboDM trajectories in {robodm_dir}")

    # Step 3: Analyze trajectories with Llama 3.2-Vision
    print("\n3. Analyzing trajectories with Llama 3.2-Vision...")
    detector = DROIDSuccessDetector()

    # Get converted trajectory paths
    success_vla_paths = sorted(Path(robodm_dir).glob("success_*.vla"))
    failure_vla_paths = sorted(Path(robodm_dir).glob("failure_*.vla"))

    # Analyze and compare
    detector.compare_trajectories(
        success_paths=[str(p) for p in success_vla_paths],
        failure_paths=[str(p) for p in failure_vla_paths],
    )

    print("\n" + "=" * 60)
    print(
        "Demo complete! The Llama 3.2-Vision model successfully analyzed DROID trajectories."
    )
    print("\nKey insights:")
    print(
        "- Llama 3.2-Vision can detect task completion indicators in robotic trajectories")
    print("- Success/failure patterns can be identified from visual analysis")
    print("- Frame-by-frame analysis provides detailed task understanding")


if __name__ == "__main__":
    main()
