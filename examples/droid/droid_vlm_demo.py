"""
Enhanced demo script using RoboDM Agent with Llama 3.2-Vision for trajectory success/failure classification.

This script demonstrates the full RoboDM Agent capabilities:
1. Downloads sample DROID trajectories (both success and failure)
2. Converts them to RoboDM format and creates Ray dataset
3. Uses Agent.filter() with natural language prompts like "trajectories that are successful"
4. Leverages planner.py to generate filter functions via LLM
5. Uses executor.py to apply filters on Ray dataset with parallel processing
6. Demonstrates robo2vlm tool for vision-language analysis

Key improvements over basic demo:
- Natural language interface: agent.filter("trajectories that are successful")
- LLM-generated code execution via planner and executor
- Parallel processing with Ray datasets
- Extensible tool system including robo2vlm
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ray
from download_droid import DROIDDownloader
from droid_to_robodm import DROIDToRoboDMConverter

import robodm
from robodm.agent import Agent


class DROIDSuccessDetector:
    """Enhanced DROID success/failure detector using RoboDM Agent system."""

    def __init__(self):
        """Initialize the detector with Agent capabilities."""

        # Configure Ray's data context for better handling of complex objects
        import ray.data
        ctx = ray.data.DataContext.get_current()
        ctx.enable_tensor_extension_casting = False
        # Use pickle for complex objects instead of Arrow
        ctx.use_push_based_shuffle = False
        
        print("Initializing RoboDM Agent with Llama 3.2-Vision...")
        
        # Configure tools for the Agent with proper vLLM settings for Llama 3.2 Vision
        self.tools_config = {
            "tools": {
                "robo2vlm": {
                    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "temperature": 0.1,
                    "max_tokens": 100,
                    # "enforce_eager": True,
                    "context_length": 1024  # Reduce memory usage
                }
            }
        }
        
        print("Agent configuration ready!")

    def create_ray_dataset(self, robodm_dir: str) -> ray.data.Dataset:
        """
        Create Ray dataset from RoboDM trajectories for Agent processing.
        
        Args:
            robodm_dir: Directory containing RoboDM trajectory files
            
        Returns:
            Ray dataset ready for Agent operations
        """
        print("Creating Ray dataset from RoboDM trajectories...")
        
        trajectory_paths = list(Path(robodm_dir).glob("*.vla"))
        dataset_items = []
        
        for traj_path in trajectory_paths:
            try:
                # Load trajectory data
                traj = robodm.Trajectory(path=str(traj_path), mode="r")
                data = traj.load()
                
                # Extract key information
                camera_keys = [k for k in data.keys() if "observation/images/" in k]
                primary_camera = camera_keys[0] if camera_keys else None
                
                # Create dataset item with Ray-compatible data types
                item = {
                    "trajectory_path": str(traj_path),
                    "trajectory_name": traj_path.stem,
                    "is_success_labeled": "success" in traj_path.stem,
                    "num_frames": len(data.get(primary_camera, [])) if primary_camera else 0,
                    # Convert list to string to avoid Arrow conversion issues
                    "camera_keys": ",".join(camera_keys) if camera_keys else "",
                    "primary_camera": primary_camera or "",
                }
                
                # Only include frames if we have valid data - convert to smaller format to avoid memory issues
                if primary_camera and len(data[primary_camera]) > 0:
                    # Store frame indices instead of actual frames to reduce memory
                    item["has_frames"] = True
                    item["first_frame_idx"] = 0
                    item["last_frame_idx"] = len(data[primary_camera]) - 1
                    item["middle_frame_idx"] = len(data[primary_camera]) // 2
                    
                    # Store actual frames as pickled objects (small subset)
                    # Convert to uint8 and ensure proper shape
                    first_frame = data[primary_camera][0]
                    if isinstance(first_frame, np.ndarray):
                        if first_frame.dtype != np.uint8:
                            first_frame = (first_frame * 255).astype(np.uint8) if first_frame.max() <= 1.0 else first_frame.astype(np.uint8)
                        item["first_frame"] = first_frame
                    else:
                        item["first_frame"] = None
                        
                    middle_frame = data[primary_camera][len(data[primary_camera])//2]
                    if isinstance(middle_frame, np.ndarray):
                        if middle_frame.dtype != np.uint8:
                            middle_frame = (middle_frame * 255).astype(np.uint8) if middle_frame.max() <= 1.0 else middle_frame.astype(np.uint8)
                        item["middle_frame"] = middle_frame
                    else:
                        item["middle_frame"] = None
                        
                    last_frame = data[primary_camera][-1]
                    if isinstance(last_frame, np.ndarray):
                        if last_frame.dtype != np.uint8:
                            last_frame = (last_frame * 255).astype(np.uint8) if last_frame.max() <= 1.0 else last_frame.astype(np.uint8)
                        item["last_frame"] = last_frame
                    else:
                        item["last_frame"] = None
                else:
                    item["has_frames"] = False
                    item["first_frame"] = None
                    item["middle_frame"] = None
                    item["last_frame"] = None
                
                dataset_items.append(item)
                traj.close()
                
            except Exception as e:
                print(f"Warning: Could not process {traj_path}: {e}")
                continue
        
        # Create dataset with explicit schema to avoid type inference issues
        dataset = ray.data.from_items(dataset_items)
        print(f"Created Ray dataset with {dataset.count()} trajectories")
        
        return dataset

    def filter_successful_trajectories(self, agent: 'Agent') -> ray.data.Dataset:
        """
        Use Agent.filter() with natural language to find successful trajectories.
        This demonstrates the planner generating filter functions and executor applying them.
        
        Args:
            agent: Agent instance to use for filtering
            
        Returns:
            Filtered dataset containing only successful trajectories
        """
        print("Using Agent.filter() with natural language...")
        
        print(f"Agent initialized with {len(agent)} trajectories")
        print(f"Available tools: {agent.list_tools()}")
        
        # Show dataset schema that planner will use
        print("Dataset schema for planner:")
        schema_info = agent.inspect_schema()
        for key in schema_info["keys"][:5]:
            print(f"  trajectory['{key}']: {type(schema_info['sample_values'].get(key, 'unknown')).__name__}")
        
        print("\nTesting Agent.filter() with natural language prompt...")
        print('Prompt: "trajectories that are successful"')
        print("This will trigger:")
        print("  1. planner.generate_filter_function() - LLM generates code")
        print("  2. executor.apply_filter() - Runs filter on Ray dataset")
        
        start_time = time.time()
        successful_trajectories = agent.filter("trajectories that are successful")
        filter_time = time.time() - start_time
        
        success_count = successful_trajectories.count()
        print(f"Filter completed: {success_count}/{len(agent)} trajectories")
        print(f"Execution time: {filter_time:.2f} seconds")

        # Debug: Inspect the structure of filtered data
        print("DEBUG: Inspecting filtered dataset structure...")
        if success_count > 0:
            sample_filtered = successful_trajectories.take(1)[0]
            print(f"Filtered dataset keys: {list(sample_filtered.keys())}")
            print(f"Sample filtered trajectory type: {type(sample_filtered)}")
        
        return successful_trajectories
    
    def analyze_with_vision_model(self, agent: Agent, trajectories: ray.data.Dataset):
        """
        Demonstrate robo2vlm tool usage through the Agent system.
        
        Args:
            agent: Agent instance with robo2vlm tool
            trajectories: Dataset to analyze
        """
        print("Analyzing trajectories with robo2vlm tool...")
        
        if trajectories.count() == 0:
            print("No trajectories to analyze")
            return
        
        # Get a sample trajectory
        sample_traj = trajectories.take(1)[0]
        print(f"Analyzing: {sample_traj.get('trajectory_name', 'unknown')}")
        
        # Get the robo2vlm tool from agent
        robo2vlm = agent.tools_manager.get_tool("robo2vlm")
        
        # Analyze key frames
        frames_to_analyze = ["first_frame", "middle_frame", "last_frame"]
        questions = [
            "Is the robot gripper holding any object? Answer yes or no.",
            "Describe what task the robot appears to be performing.",
            "Are there any signs of task completion or success?"
        ]
        
        for frame_name in frames_to_analyze:
            frame = sample_traj.get(frame_name)
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                print(f"\nAnalyzing {frame_name}:")
                try:
                    for question in questions:
                        response = robo2vlm(frame, question)
                        print(f"  Q: {question}")
                        print(f"  A: {response}")
                except Exception as e:
                    print(f"  Error analyzing {frame_name}: {e}")
            else:
                print(f"\nSkipping {frame_name}: No valid frame data available")

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

    def compare_trajectories_with_agent(self, dataset: ray.data.Dataset):
        """
        Compare trajectories using Agent system with natural language filtering.
        
        Args:
            dataset: Ray dataset containing all trajectories
        """
        print("\n" + "=" * 60)
        print("AGENT-BASED TRAJECTORY ANALYSIS")
        print("=" * 60)
        
        # Create Agent with memory-optimized configuration (single instance)
        agent = Agent(dataset, 
                     llm_model="Qwen/Qwen2.5-VL-3B-Instruct",
                     tools_config=self.tools_config,
                     context_length=1024)
        
        print(f"Analyzing {len(agent)} trajectories with Agent system")
        
        # Filter successful trajectories using natural language (reuse agent)
        successful_trajectories = self.filter_successful_trajectories(agent)
        
        # Analyze with vision model
        self.analyze_with_vision_model(agent, successful_trajectories)
        
        # Demonstrate other Agent capabilities
        print("\n--- ADDITIONAL AGENT CAPABILITIES ---")
        
        print("\nTesting agent.map() for trajectory enhancement...")
        enhanced_dataset = agent.map("add basic statistics and frame analysis")
        print(f"Map operation result: {enhanced_dataset.count()} enhanced trajectories")
        
        # print("\nTesting agent.analyze() for dataset insights...")
        # analysis_result = agent.analyze("what is the success rate and common patterns?")
        # print(f"Analysis result: {analysis_result}")
        
        # Show classification results
        print("\n--- CLASSIFICATION RESULTS ---")
        total_trajectories = dataset.count()
        successful_count = successful_trajectories.count()
        
        print(f"Total trajectories: {total_trajectories}")
        print(f"Classified as successful: {successful_count}")
        print(f"Classified as failed: {total_trajectories - successful_count}")
        
        # Show ground truth comparison
        labeled_success = dataset.filter(lambda x: x["is_success_labeled"]).count()
        labeled_failure = total_trajectories - labeled_success
        
        print(f"\nGround truth (from labels):")
        print(f"  Successful: {labeled_success}")
        print(f"  Failed: {labeled_failure}")
        
        # --- Accuracy computation ---
        # Fix the KeyError by properly handling filtered dataset structure
        print("\nDEBUG: Computing accuracy...")
        try:
            gt_records = dataset.take(total_trajectories)
            print(f"Original dataset sample keys: {list(gt_records[0].keys()) if gt_records else 'No data'}")
            
            # Get predicted successful trajectories - handle potential key differences
            if successful_count > 0:
                try:
                    pred_trajectories = successful_trajectories.take(successful_count)
                    print(f"Filtered dataset sample keys: {list(pred_trajectories[0].keys()) if pred_trajectories else 'No data'}")
                    
                    # Build prediction set using available key (might be different after filtering)
                    pred_success_paths = set()
                    for traj in pred_trajectories:
                        # Try different possible keys that might contain the path
                        path_key = None
                        for key in ["trajectory_path", "path", "trajectory_name", "name"]:
                            if key in traj:
                                path_key = key
                                break
                        
                        if path_key:
                            pred_success_paths.add(traj[path_key])
                        else:
                            print(f"Warning: No path identifier found in filtered trajectory. Available keys: {list(traj.keys())}")
                except Exception as e:
                    print(f"Error processing filtered trajectories: {e}")
                    pred_success_paths = set()
            else:
                pred_success_paths = set()
            
            print(f"Predicted successful paths: {pred_success_paths}")
            
            correct = 0
            for traj in gt_records:
                gt_success = traj["is_success_labeled"]
                # Use the same key to match against predictions
                traj_identifier = traj.get("trajectory_path") or traj.get("trajectory_name", "unknown")
                pred_success = traj_identifier in pred_success_paths
                if gt_success == pred_success:
                    correct += 1

            accuracy = correct / total_trajectories if total_trajectories else 0.0

            print(f"\nPrediction accuracy: {accuracy:.2%}  ( {correct}/{total_trajectories} correct )")
        except Exception as e:
            print(f"Error computing accuracy: {e}")
            print("Skipping accuracy computation")

        return agent


def main():
    """Enhanced main demo function using RoboDM Agent system."""
    print("Enhanced DROID Trajectory Success/Failure Detection with RoboDM Agent")
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

    # Step 3: Create Ray dataset and analyze with Agent
    print("\n3. Creating Ray dataset and initializing Agent...")
    detector = DROIDSuccessDetector()
    
    # Create Ray dataset from trajectories
    dataset = detector.create_ray_dataset(robodm_dir)
    
    # Use Agent system for analysis
    agent = detector.compare_trajectories_with_agent(dataset)
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
