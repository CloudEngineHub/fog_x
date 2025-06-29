"""
Simplified demo script using robo2vlm tool to classify DROID trajectories.

This version uses a mock VLM for demonstration purposes when the actual model is not available.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import robodm
from robodm.agent.tools import ToolsManager, create_vision_config
from download_droid import DROIDDownloader
from droid_to_robodm import DROIDToRoboDMConverter


class MockVLMTool:
    """Mock VLM tool for demonstration when actual model is not available."""
    
    def __call__(self, frame: np.ndarray, prompt: str) -> str:
        """Simulate VLM responses based on trajectory characteristics."""
        # Simple heuristics based on frame statistics
        mean_intensity = np.mean(frame)
        std_intensity = np.std(frame)
        
        if "gripper holding" in prompt.lower():
            # Higher intensity variance might indicate object presence
            if std_intensity > 30:
                return "Yes, the gripper appears to be holding an object."
            else:
                return "No, the gripper appears to be empty."
        
        elif "task" in prompt.lower() and "performing" in prompt.lower():
            # Simulate task descriptions
            if mean_intensity > 100:
                return "The robot appears to be performing a pick and place task."
            else:
                return "The robot appears to be reaching or grasping."
        
        elif "failure" in prompt.lower() or "signs" in prompt.lower():
            # Low variance might indicate stuck robot
            if std_intensity < 20:
                return "Yes, the robot appears to be stuck or stationary."
            else:
                return "No visible signs of failure."
        
        elif "completed successfully" in prompt.lower():
            # Higher mean intensity might indicate success
            if mean_intensity > 120:
                return "Yes, the task appears completed."
            else:
                return "No, the task is still in progress."
        
        return "Unable to determine from this frame."


class DROIDSuccessDetector:
    """Detect success/failure in DROID trajectories using VLM."""
    
    def __init__(self, use_mock=False):
        if use_mock:
            print("Using mock VLM for demonstration")
            self.vlm_tool = MockVLMTool()
        else:
            # Try to use actual VLM tool
            try:
                self.manager = ToolsManager(config=create_vision_config())
                self.vlm_tool = self.manager.get_tool("robo2vlm")
                print("Using actual robo2vlm tool")
            except Exception as e:
                print(f"Could not load actual VLM, using mock: {e}")
                self.vlm_tool = MockVLMTool()
        
    def analyze_trajectory_frames(self, trajectory_path: str, sample_rate: int = 50) -> Dict:
        """
        Analyze frames from a trajectory using VLM.
        
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
            "overall_assessment": None
        }
        
        if not camera_keys:
            print(f"No camera data found in {trajectory_path}")
            return results
        
        # Use the first available camera
        primary_camera = camera_keys[0]
        frames = data[primary_camera]
        
        print(f"  Analyzing {len(frames)} frames from {primary_camera} (sampling every {sample_rate} frames)")
        
        # Sample frames for analysis
        frame_indices = list(range(0, len(frames), sample_rate))[:5]  # Limit to 5 frames for demo
        
        for i, idx in enumerate(frame_indices):
            frame = frames[idx]
            print(f"  Analyzing frame {i+1}/{len(frame_indices)}...")
            
            # Analyze frame for task completion indicators
            prompts = [
                "Is the robot gripper holding any object?",
                "Describe what task the robot appears to be performing.",
                "Are there any signs of failure?",
                "Is the task completed successfully in this frame?"
            ]
            
            frame_analysis = {
                "frame_idx": idx,
                "analyses": {}
            }
            
            for prompt in prompts:
                try:
                    response = self.vlm_tool(frame, prompt)
                    frame_analysis["analyses"][prompt] = response
                except Exception as e:
                    print(f"    Error with prompt '{prompt}': {e}")
                    frame_analysis["analyses"][prompt] = "Error"
            
            results["frame_analyses"].append(frame_analysis)
        
        # Analyze trajectory progression
        results["overall_assessment"] = self._assess_trajectory_success(results["frame_analyses"])
        
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
            if "yes" in responses.get("Is the robot gripper holding any object?", "").lower():
                success_indicators += 1
            
            # Check for failure signs
            failure_response = responses.get("Are there any signs of failure?", "")
            if "yes" in failure_response.lower():
                failure_indicators += 1
            
            # Check for task completion
            if "yes" in responses.get("Is the task completed successfully in this frame?", "").lower():
                success_indicators += 1
            
            # Collect task descriptions
            task_desc = responses.get("Describe what task the robot appears to be performing.", "")
            if task_desc and task_desc != "Error":
                task_descriptions.append(task_desc)
        
        # Determine overall success
        total_frames = len(frame_analyses)
        success_rate = success_indicators / (total_frames * 2) if total_frames > 0 else 0
        failure_rate = failure_indicators / total_frames if total_frames > 0 else 0
        
        is_successful = success_rate > 0.3 and failure_rate < 0.3
        
        return {
            "is_successful": is_successful,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "success_indicators": success_indicators,
            "failure_indicators": failure_indicators,
            "common_task": max(set(task_descriptions), key=task_descriptions.count) if task_descriptions else "Unknown"
        }
    
    def compare_trajectories(self, success_paths: List[str], failure_paths: List[str]):
        """
        Compare successful and failed trajectories.
        
        Args:
            success_paths: List of successful trajectory paths
            failure_paths: List of failed trajectory paths
        """
        print("\n" + "="*60)
        print("TRAJECTORY ANALYSIS RESULTS")
        print("="*60)
        
        # Analyze successful trajectories
        print("\n--- LABELED SUCCESSFUL TRAJECTORIES ---")
        success_results = []
        for path in success_paths:
            if os.path.exists(path):
                print(f"\nAnalyzing: {os.path.basename(path)}")
                result = self.analyze_trajectory_frames(path)
                success_results.append(result)
                
                assessment = result["overall_assessment"]
                print(f"  VLM Prediction: {'SUCCESS' if assessment['is_successful'] else 'FAILURE'}")
                print(f"  Success indicators: {assessment['success_indicators']}")
                print(f"  Failure indicators: {assessment['failure_indicators']}")
                print(f"  Common task: {assessment['common_task']}")
        
        # Analyze failed trajectories
        print("\n--- LABELED FAILED TRAJECTORIES ---")
        failure_results = []
        for path in failure_paths:
            if os.path.exists(path):
                print(f"\nAnalyzing: {os.path.basename(path)}")
                result = self.analyze_trajectory_frames(path)
                failure_results.append(result)
                
                assessment = result["overall_assessment"]
                print(f"  VLM Prediction: {'SUCCESS' if assessment['is_successful'] else 'FAILURE'}")
                print(f"  Success indicators: {assessment['success_indicators']}")
                print(f"  Failure indicators: {assessment['failure_indicators']}")
                print(f"  Common task: {assessment['common_task']}")
        
        # Calculate accuracy
        print("\n--- CLASSIFICATION ACCURACY ---")
        correct_success = sum(1 for r in success_results if r["overall_assessment"]["is_successful"])
        correct_failure = sum(1 for r in failure_results if not r["overall_assessment"]["is_successful"])
        total_success = len(success_results)
        total_failure = len(failure_results)
        
        if total_success > 0:
            success_accuracy = correct_success / total_success
            print(f"Success detection accuracy: {success_accuracy:.0%} ({correct_success}/{total_success})")
        
        if total_failure > 0:
            failure_accuracy = correct_failure / total_failure
            print(f"Failure detection accuracy: {failure_accuracy:.0%} ({correct_failure}/{total_failure})")
        
        if total_success + total_failure > 0:
            overall_accuracy = (correct_success + correct_failure) / (total_success + total_failure)
            print(f"Overall accuracy: {overall_accuracy:.0%}")


def main():
    """Main demo function."""
    print("DROID Trajectory Success/Failure Detection Demo")
    print("=" * 60)
    
    # Check if data already exists
    robodm_dir = "./robodm_trajectories"
    if not os.path.exists(robodm_dir):
        print("\nPlease run the following commands first:")
        print("1. python download_droid.py")
        print("2. python droid_to_robodm.py")
        return
    
    # Step 3: Analyze trajectories with VLM
    print("\nAnalyzing trajectories with robo2vlm...")
    detector = DROIDSuccessDetector(use_mock=True)  # Use mock for demo
    
    # Get converted trajectory paths
    success_vla_paths = sorted(Path(robodm_dir).glob("success_*.vla"))[:2]
    failure_vla_paths = sorted(Path(robodm_dir).glob("failure_*.vla"))[:2]
    
    print(f"Found {len(success_vla_paths)} successful and {len(failure_vla_paths)} failed trajectories")
    
    # Analyze and compare
    detector.compare_trajectories(
        success_paths=[str(p) for p in success_vla_paths],
        failure_paths=[str(p) for p in failure_vla_paths]
    )
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("\nThis demo shows how the robo2vlm tool can be used to:")
    print("- Analyze individual frames from robot trajectories")
    print("- Detect task completion indicators")
    print("- Classify trajectories as successful or failed")
    print("- Extract common task patterns from visual data")


if __name__ == "__main__":
    main()