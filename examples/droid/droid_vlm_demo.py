"""
Enhanced demo script using RoboDM Agent with VLM for trajectory success/failure classification.

This script demonstrates the full RoboDM Agent capabilities:
1. Downloads sample DROID trajectories (both success and failure)
2. Creates a proper VLADataset from file paths (not pre-loaded data)
3. Uses load_trajectories() for parallel loading
4. Demonstrates filter execution with Executor (bypassing planner for now)
5. Shows how VLM tools can be used during filtering
"""

# python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-VL-32B-Instruct   --host 0.0.0.0   --port 30000 --tp 8 

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
import ray

import robodm
from robodm.dataset import VLADataset, DatasetConfig
from robodm.agent import Agent
from robodm.agent.executor import Executor
from robodm.agent.tools import ToolsManager


class DROIDSuccessDetector:
    """Enhanced DROID success/failure detector using RoboDM Agent system."""

    def __init__(self, max_trajectories: Optional[int] = None):
        """Initialize the detector with Agent capabilities.
        
        Args:
            max_trajectories: Maximum number of trajectories to process. If None, processes all trajectories.
        """
        print("Initializing RoboDM Agent with VLM tools...")
        
        self.max_trajectories = max_trajectories
        if max_trajectories is not None:
            print(f"Will limit processing to maximum {max_trajectories} trajectories")
        
        # Configure tools for the Agent
        self.tools_config = {
            "tools": {
                "robo2vlm": {
                    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "context_length": 1024
                }
            }
        }
        
        # Initialize tools manager
        self.tools_manager = ToolsManager(config=self.tools_config)
        
        # Initialize executor with tools
        self.executor = Executor(tools_manager=self.tools_manager)
        
        print("Agent configuration ready!")

    def create_robodm_dataset(self, robodm_dir: str) -> VLADataset:
        """
        Create VLADataset from RoboDM trajectory files.
        
        This properly uses VLADataset to start with file paths and enable
        lazy loading with load_trajectories().
        
        Args:
            robodm_dir: Directory containing RoboDM trajectory files
            
        Returns:
            VLADataset ready for parallel processing
        """
        print("Creating VLADataset from RoboDM trajectories...")
        
        # Configure dataset for parallel loading
        config = DatasetConfig(
            batch_size=4,
            shuffle=False,
            use_metadata=True,
            auto_build_metadata=False  # We'll manage metadata manually for now
        )
        
        # Create VLADataset from directory
        # This creates a Ray dataset with just file paths
        dataset = VLADataset(
            path=robodm_dir,
            return_type="numpy",
            config=config
        )
        
        total_trajectories = dataset.count()
        print(f"Found {total_trajectories} trajectory files")
        
        # Apply max_trajectories limit if specified
        if self.max_trajectories is not None and total_trajectories > self.max_trajectories:
            print(f"Limiting to {self.max_trajectories} trajectories (out of {total_trajectories} total)")
            # Use take() to limit the number of trajectories
            limited_items = dataset.take(self.max_trajectories)
            
            # Create a new VLADataset from the limited items
            # We need to extract file paths from the limited items
            if limited_items:
                # Extract file paths from the limited items 
                # The items are currently just string paths from the Ray dataset
                limited_file_paths = [item if isinstance(item, str) else item.get("item", str(item)) 
                                    for item in limited_items]
                
                # Create a new VLADataset with limited file paths
                import ray.data as rd
                limited_ray_dataset = rd.from_items(limited_file_paths)
                if config.shuffle:
                    limited_ray_dataset = limited_ray_dataset.random_shuffle()
                
                # Create new VLADataset instance with limited data
                limited_dataset = VLADataset.__new__(VLADataset)
                limited_dataset.path = dataset.path
                limited_dataset.return_type = dataset.return_type
                limited_dataset.config = dataset.config
                limited_dataset.file_paths = limited_file_paths
                limited_dataset.ray_dataset = limited_ray_dataset
                limited_dataset.metadata_manager = dataset.metadata_manager
                limited_dataset._schema = None
                limited_dataset._stats = None
                limited_dataset._is_loaded = False
                limited_dataset._has_file_paths = True
                
                dataset = limited_dataset
                print(f"Limited dataset created with {dataset.count()} trajectory files")
        else:
            print(f"Processing all {total_trajectories} trajectory files")
        
        print(f"Dataset type: {type(dataset)}")
        print(f"Has _is_loaded: {hasattr(dataset, '_is_loaded')}")
        print(f"Is loaded: {dataset._is_loaded}")
        
        return dataset

    def calculate_trajectory_captioning_accuracy(self, dataset: VLADataset):
        """
        Calculate accuracy for trajectory captioning by comparing VLM-generated captions
        with ground truth language descriptions from metadata using LLM for semantic matching.
        
        Args:
            dataset: VLADataset with loaded trajectories
            
        Returns:
            float: Accuracy of caption matching
        """
        print("\n" + "=" * 60)
        print("TRAJECTORY CAPTIONING ACCURACY CALCULATION")
        print("=" * 60)
        
        # Create output directory for captioning results
        caption_output_dir = Path("./trajectory_captioning_results")
        caption_output_dir.mkdir(exist_ok=True)
        
        def extract_caption_and_description(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            """Extract VLM caption and ground truth description from trajectory."""
            import json
            from pathlib import Path
            import numpy as np
            import cv2
            
            file_path = trajectory.get("__file_path__", "")
            traj_name = Path(file_path).stem
            
            # Only process successful trajectories
            if "success" not in file_path.lower():
                return {
                    "trajectory_name": traj_name,
                    "ground_truth_description": "",
                    "vlm_caption": "",
                    "has_ground_truth": False,
                    "has_caption": False,
                    "is_match": False,
                    "comparison_explanation": "Skipped - not a successful trajectory"
                }
            
            # Parse metadata to get language description
            ground_truth_description = ""
            try:
                metadata_data = trajectory.get("metadata", None)
                if metadata_data is not None:
                    # Handle case where metadata is stored as a numpy array/list from trajectory loading
                    if isinstance(metadata_data, (list, np.ndarray)) and len(metadata_data) > 0:
                        metadata_str = metadata_data[0]
                    else:
                        metadata_str = metadata_data
                    
                    # Parse the JSON string
                    if metadata_str:
                        metadata = json.loads(metadata_str)
                        # Get language instruction from metadata
                        # Use current_task as it contains the task description in DROID dataset
                        ground_truth_description = metadata.get("current_task", "")
                        
                        # If current_task is not available, try language_instruction fields
                        if not ground_truth_description:
                            ground_truth_description = (
                                metadata.get("language_instruction", "") or
                                metadata.get("language_instruction_2", "") or
                                metadata.get("language_instruction_3", "")
                            )
            except Exception as e:
                print(f"Error parsing metadata for {traj_name}: {e}")
                import traceback
                traceback.print_exc()
                
            
            # Get VLM caption
            vlm_caption = ""
            try:
                # Find camera keys
                camera_keys = [k for k in trajectory.keys() 
                             if "observation/images/" in k or "image" in k.lower()]
                
                if camera_keys:
                    primary_camera = camera_keys[3] if len(camera_keys) > 1 else camera_keys[0]
                    frames = trajectory.get(primary_camera, [])
                    
                    if len(frames) >= 8:
                        # Extract frames evenly distributed throughout the trajectory
                        num_frames = 6  # Extract 6 frames for captioning
                        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
                        selected_frames = [frames[i] for i in indices]
                        
                        # Create 2x3 grid for better trajectory understanding
                        # Use original frame sizes without resizing
                        
                        # Create 2x3 grid
                        top_row = np.hstack(selected_frames[:3])
                        bottom_row = np.hstack(selected_frames[3:])
                        stitched_frame = np.vstack([top_row, bottom_row])
                        
                        # Save input image
                        image_filename = caption_output_dir / f"{traj_name}_caption_input.jpg"
                        cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                        
                        # Use VLM to generate caption
                        from robodm.agent.vlm_service import get_vlm_service
                        vlm_service = get_vlm_service()
                        vlm_service.initialize()
                        
                        vlm_prompt = (
                            "These are 6 frames from a robot trajectory shown in temporal order "
                            "(left to right, top to bottom). Please describe with one sentence what task the robot "
                            "is performing in this trajectory. Be very specific about the "
                            "actions and objects involved."
                        )
                        vlm_caption = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                        
                        print(f"📝 Captioning {traj_name}")
                        print(f"   GT: '{ground_truth_description}...'")
                        print(f"   VLM: '{vlm_caption}...'")
                        
                    else:
                        print(f"⚠️ Trajectory {traj_name} has only {len(frames)} frames, skipping captioning")
                        
            except Exception as e:
                print(f"Error generating VLM caption for {traj_name}: {e}")
                import traceback
                traceback.print_exc()
            
            # Use LLM to compare descriptions semantically
            is_match = False
            comparison_explanation = ""
            
            if ground_truth_description and vlm_caption:
                try:
                    from robodm.agent.vlm_service import get_vlm_service
                    vlm_service = get_vlm_service()
                    
                    comparison_prompt = f"""Compare these two robot task descriptions and determine if they describe the same or similar task:

Description 1 (Ground Truth): {ground_truth_description}

Description 2 (VLM Caption): {vlm_caption}

Be generous in your matching. Only say NO if they describe COMPLETELY different tasks with different goals.
It is fine that the VLM Caption is more specific compared to the Ground Truth.

Respond with only YES or NO followed by a brief explanation.

Format:
YES/NO: Your one sentence explanation"""

                    comparison_response = vlm_service.generate_code(comparison_prompt)
                    
                    # Parse the response
                    response_lower = comparison_response.strip().lower()
                    if response_lower.startswith("yes"):
                        is_match = True
                        comparison_explanation = comparison_response[3:].strip(": ")
                    elif response_lower.startswith("no"):
                        is_match = False
                        comparison_explanation = comparison_response[2:].strip(": ")
                    else:
                        # Try to find YES or NO in the response
                        is_match = "yes" in response_lower.split()[0:3]
                        comparison_explanation = comparison_response
                    
                    print(f"   Match: {'YES' if is_match else 'NO'}")
                    
                except Exception as e:
                    print(f"Error comparing descriptions: {e}")
                    comparison_explanation = f"Error: {str(e)}"
            
            # Save results
            results_filename = caption_output_dir / f"{traj_name}_caption_results.txt"
            with open(results_filename, 'w') as f:
                f.write(f"Trajectory Captioning Results\n")
                f.write(f"============================\n")
                f.write(f"Trajectory: {traj_name}\n")
                f.write(f"File path: {file_path}\n")
                f.write(f"\nGround Truth Description:\n{ground_truth_description}\n")
                f.write(f"\nVLM Generated Caption:\n{vlm_caption}\n")
                f.write(f"\nSemantic Comparison:\n")
                f.write(f"Match: {'YES' if is_match else 'NO'}\n")
                f.write(f"Explanation: {comparison_explanation}\n")
                f.write(f"\nInput image saved as: {traj_name}_caption_input.jpg\n")
            
            return {
                "trajectory_name": traj_name,
                "ground_truth_description": ground_truth_description,
                "vlm_caption": vlm_caption,
                "has_ground_truth": bool(ground_truth_description),
                "has_caption": bool(vlm_caption),
                "is_match": is_match,
                "comparison_explanation": comparison_explanation
            }
        
        # Apply transformation to get all captions
        results_dataset = dataset.map(extract_caption_and_description).materialize()
        results = list(results_dataset.iter_rows())
        
        # Calculate accuracy based on LLM matching
        correct_matches = 0  # Number of correct caption matches
        valid_comparisons = 0
        skipped_trajectories = 0
        
        print("\nDetailed Caption Comparison Results:")
        print("-" * 80)
        
        for result in results:
            if not result["has_ground_truth"] and not result["has_caption"] and "Skipped" in result.get("comparison_explanation", ""):
                skipped_trajectories += 1
                continue
                
            if result["has_ground_truth"] and result["has_caption"]:
                valid_comparisons += 1
                
                # Get the match result
                is_match = result["is_match"]
                
                # Count correct matches (we expect captions to match ground truth)
                if is_match:
                    correct_matches += 1
                
                status = "✅" if is_match else "❌"
                print(f"{status} {result['trajectory_name']}: {'MATCH' if is_match else 'NO MATCH'}")
                print(f"   Explanation: {result['comparison_explanation']}")
                print()
        
        # Calculate accuracy
        if valid_comparisons > 0:
            accuracy = correct_matches / valid_comparisons
        else:
            accuracy = 0
            print("⚠️ No valid comparisons found (missing ground truth or captions)")
        
        print(f"\nOverall Captioning Metrics:")
        print(f"Total trajectories: {len(results)}")
        print(f"Successful trajectories processed: {valid_comparisons}")
        print(f"Failed trajectories skipped: {skipped_trajectories}")
        print(f"Correct matches: {correct_matches}")
        print(f"Incorrect matches: {valid_comparisons - correct_matches}")
        print(f"Accuracy: {accuracy:.3f} ({correct_matches}/{valid_comparisons})")
        
        # Summary of results
        summary_filename = caption_output_dir / "captioning_accuracy_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"Trajectory Captioning Accuracy Summary\n")
            f.write(f"=====================================\n")
            f.write(f"Total trajectories: {len(results)}\n")
            f.write(f"Successful trajectories processed: {valid_comparisons}\n")
            f.write(f"Failed trajectories skipped: {skipped_trajectories}\n")
            f.write(f"Correct matches: {correct_matches}\n")
            f.write(f"Incorrect matches: {valid_comparisons - correct_matches}\n")
            f.write(f"Accuracy: {accuracy:.3f} ({correct_matches}/{valid_comparisons})\n")
        
        print(f"\n✅ Results saved to {caption_output_dir}/")
        
        return accuracy

    def calculate_f1_matrix(self, dataset: VLADataset):
        """
        Calculate and print F1 matrix by comparing ground truth labels with VLM predictions.
        
        Args:
            dataset: VLADataset with loaded trajectories
        """
        print("\n" + "=" * 60)
        print("F1 MATRIX CALCULATION")
        print("=" * 60)
        
        # Create output directory for F1 matrix results
        f1_output_dir = Path("./f1_matrix_results")
        f1_output_dir.mkdir(exist_ok=True)
        
        # Transform to extract labels and predictions
        def extract_labels_and_predictions(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            """Extract ground truth and VLM predictions for F1 calculation with file saving."""
            from pathlib import Path
            import numpy as np
            import cv2
            
            file_path = trajectory.get("__file_path__", "")
            ground_truth = "success" in file_path.lower()
            traj_name = Path(file_path).stem
            
            # Get VLM prediction and save all results
            vlm_prediction = False
            vlm_response = "No VLM analysis performed"
            
            try:
                # Find camera keys
                camera_keys = [k for k in trajectory.keys() 
                             if "observation/images/" in k or "image" in k.lower()]
                print(f"Camera keys: {camera_keys}")
                
                if camera_keys:
                    primary_camera = camera_keys[3] if len(camera_keys) > 1 else camera_keys[0]
                    frames = trajectory.get(primary_camera, [])
                    print(f"Frames: {len(frames)}, {frames[0].shape}")
                    
                    if len(frames) >= 4:
                        # Select 4 frames: start, 1/3, 2/3, and end
                        indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
                        selected_frames = [frames[i] for i in indices]
                        
                        # Create 2x2 grid
                        h, w = selected_frames[0].shape[:2]
                        resized_frames = []
                        for frame in selected_frames:
                            if frame.shape[:2] != (h, w):
                                frame = cv2.resize(frame, (w, h))
                            resized_frames.append(frame)
                        
                        top_row = np.hstack([resized_frames[0], resized_frames[1]])
                        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
                        stitched_frame = np.vstack([top_row, bottom_row])
                        
                        # Save input image
                        image_filename = f1_output_dir / f"{traj_name}_input.jpg"
                        cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                        
                        # Use VLM to get prediction
                        from robodm.agent.vlm_service import get_vlm_service
                        vlm_service = get_vlm_service()
                        vlm_service.initialize()
                        
                        vlm_prompt = "These are 4 frames from a robot trajectory. Does this trajectory look successful? First answer yes or no, then explain why."
                        vlm_response = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                        vlm_prediction = "yes" in vlm_response.lower()
                        
                        print(f"🔍 F1 Analysis for {traj_name}: GT={ground_truth}, VLM={vlm_prediction}")
                        
                    elif len(frames) > 0:
                        # If fewer than 4 frames, just use the last frame
                        stitched_frame = frames[-1]
                        
                        # Save input image
                        image_filename = f1_output_dir / f"{traj_name}_input.jpg"
                        cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                        
                        # Use VLM to get prediction
                        from robodm.agent.vlm_service import get_vlm_service
                        vlm_service = get_vlm_service()
                        vlm_service.initialize()
                        
                        vlm_prompt = "This is the final frame from a robot trajectory. Does this trajectory look successful? Answer yes or no."
                        vlm_response = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                        vlm_prediction = "yes" in vlm_response.lower()
                        
                        print(f"🔍 F1 Analysis for {traj_name}: GT={ground_truth}, VLM={vlm_prediction}")
                        
            except Exception as e:
                print(f"Error in VLM prediction for {traj_name}: {e}")
                vlm_prediction = ground_truth
                vlm_response = f"Error occurred: {str(e)}"
            
            # Save results to file
            results_filename = f1_output_dir / f"{traj_name}_results.txt"
            with open(results_filename, 'w') as f:
                f.write(f"F1 Matrix Calculation Results\n")
                f.write(f"=============================\n")
                f.write(f"Trajectory: {traj_name}\n")
                f.write(f"File path: {file_path}\n")
                f.write(f"Ground truth (success): {ground_truth}\n")
                f.write(f"VLM prediction (success): {vlm_prediction}\n")
                f.write(f"Prediction correct: {ground_truth == vlm_prediction}\n")
                f.write(f"\nVLM Prompt:\n{vlm_prompt if 'vlm_prompt' in locals() else 'No prompt used'}\n")
                f.write(f"\nVLM Response:\n{vlm_response}\n")
                f.write(f"\nInput image saved as: {traj_name}_input.jpg\n")
            
            return {
                "trajectory_name": traj_name,
                "ground_truth": ground_truth,
                "vlm_prediction": vlm_prediction,
                "vlm_response": vlm_response
            }
        
        # Apply transformation to get all predictions using VLADataset's map
        # This will automatically handle lazy loading
        results_dataset = dataset.map(extract_labels_and_predictions).materialize()
        results = list(results_dataset.iter_rows())
        
        # Calculate confusion matrix
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for result in results:
            gt = result["ground_truth"]
            pred = result["vlm_prediction"]
            
            if gt and pred:
                true_positives += 1
            elif not gt and not pred:
                true_negatives += 1
            elif not gt and pred:
                false_positives += 1
            elif gt and not pred:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(results)
        
        print(f"\nDetailed Results:")
        for result in results:
            status = "✅" if result["ground_truth"] == result["vlm_prediction"] else "❌"
            print(f"{status} {result['trajectory_name']}: GT={result['ground_truth']}, Pred={result['vlm_prediction']}")
    
    
        # Print F1 Matrix
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Fail  Success")
        print(f"Actual   Fail    {true_negatives:4d}  {false_positives:7d}")
        print(f"         Success {false_negatives:4d}  {true_positives:7d}")
        
        print(f"\nMetrics:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1_score:.3f}")
        

        
        return f1_score


def main():
    """Enhanced main demo function using proper VLADataset and Agent system."""
    print("RoboDM VLADataset and Agent Demo")
    print("=" * 60)

    # Configuration
    parser = argparse.ArgumentParser(description="Run the DROID VLM demo")
    parser.add_argument("--data_dir", type=str, default="./robodm_trajectories", help="Directory containing RoboDM trajectory files")
    parser.add_argument("--max_trajectories", type=int, default=100, help="Maximum number of trajectories to process")
    args = parser.parse_args()

    robodm_dir = args.data_dir
    max_trajectories = args.max_trajectories
    
    print(f"Configuration:")
    print(f"  Data directory: {robodm_dir}")
    print(f"  Max trajectories: {max_trajectories if max_trajectories is not None else 'All'}")
    
    # Step 3: Create VLADataset (with file paths only)
    print("\n3. Creating VLADataset...")
    detector = DROIDSuccessDetector(max_trajectories=max_trajectories)
    dataset = detector.create_robodm_dataset(robodm_dir)
    
    # # Step 5: Calculate F1 Matrix
    # print("\n5. Calculating F1 Matrix...")
    # detector.calculate_f1_matrix(dataset)
    
    # Step 6: Calculate Trajectory Captioning Accuracy
    print("\n6. Calculating Trajectory Captioning Accuracy...")
    captioning_accuracy = detector.calculate_trajectory_captioning_accuracy(dataset)
    print(f"\nFinal Trajectory Captioning Accuracy: {captioning_accuracy:.3f}")
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()