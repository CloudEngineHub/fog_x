"""
Benchmark for trajectory captioning using VLM on DROID dataset.

This script evaluates the accuracy of VLM-generated captions against ground truth
language descriptions from the DROID dataset metadata.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import cv2
import ray

from robodm.dataset import VLADataset, DatasetConfig
from robodm.agent.vlm_service import get_vlm_service


def process_single_trajectory_for_captioning(trajectory: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Standalone function to process a single trajectory for captioning evaluation.
    This is outside the class to avoid serialization issues with Ray.
    
    Args:
        trajectory: Loaded trajectory data
        output_dir: Directory to save results
        
    Returns:
        Dictionary with captioning results
    """
    file_path = trajectory.get("__file_path__", "")
    traj_name = Path(file_path).stem
    
    # Only process successful trajectories
    
    print(f"📝 Processing {traj_name}")
    
    # Extract ground truth description
    ground_truth = ""
    possible_keys = []
    
    keys = trajectory.keys()
    key_candidates = [
                    "tfds/language_instruction",
                    "tfds/language_instruction_2",
                    "tfds/language_instruction_3"
                ]
    
    
    try:        
        # Look for language instruction keys directly in the trajectory
        found_instructions = []
        
        for key in key_candidates:
            value = trajectory.get(key, "")
            
            # Check if value exists and has content
            has_content = False
            value_str = ""
            
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    value_str = str(value[0])
                    has_content = bool(value_str.strip())
            elif isinstance(value, str):
                value_str = value
                has_content = bool(value_str.strip())
            elif value:  # For other types
                value_str = str(value)
                has_content = bool(value_str.strip())
            
            if has_content:
                possible_keys.append(f"{key}: {value_str}")
                found_instructions.append(value_str)
                print(key, value_str)
        
        # Combine all found instructions into ground truth
        if found_instructions:
            # Join all instructions with semicolons
            ground_truth = "; ".join(found_instructions)
        else:
            ground_truth = ""
    except Exception as e:
        print(f"Error getting language instructions: {e}")
    
    # Generate VLM caption
    vlm_caption = ""
    try:
        # Initialize VLM service locally
        vlm_service = get_vlm_service()
        vlm_service.initialize()
        
        # Find camera keys
        camera_keys = []
        for key in trajectory.keys():
            if "raw/images/" in key or "observation/images/" in key or "image" in key.lower():
                camera_keys.append(key)
        
        if camera_keys:
            # Use wrist camera if available
            primary_camera = None
            for cam_key in camera_keys:
                if "wrist" in cam_key:
                    primary_camera = cam_key
                    break
            if primary_camera is None:
                primary_camera = camera_keys[0]
            
            frames = trajectory.get(primary_camera, [])
            
            if len(frames) >= 6:
                # Extract 6 frames evenly distributed
                num_frames = 6
                indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
                selected_frames = [frames[i] for i in indices]
                
                # Create 2x3 grid
                top_row = np.hstack(selected_frames[:3])
                bottom_row = np.hstack(selected_frames[3:])
                stitched_frame = np.vstack([top_row, bottom_row])
                
                # Save input image
                image_filename = output_dir / f"{traj_name}_caption_input.jpg"
                cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                
                # Generate caption
                vlm_prompt = (
                    "These are 6 frames from a robot trajectory shown in temporal order "
                    "(left to right, top to bottom). Please describe with one sentence what task the robot "
                    "is performing in this trajectory. Be very specific about the "
                    "actions and objects involved."
                )
                
                vlm_caption = vlm_service.analyze_image(stitched_frame, vlm_prompt)
    
    except Exception as e:
        print(f"Error generating caption for {traj_name}: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare descriptions
    is_match = False
    explanation = ""
    
    if ground_truth and vlm_caption:
        try:
            # Initialize VLM service for comparison
            vlm_service = get_vlm_service()
            vlm_service.initialize()
            
            comparison_prompt = f"""Compare these two robot task descriptions and determine if they describe the same or similar task:

Description 1 (Ground Truth): {ground_truth}

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
                explanation = comparison_response[3:].strip(": ")
            elif response_lower.startswith("no"):
                is_match = False
                explanation = comparison_response[2:].strip(": ")
            else:
                # Try to find YES or NO in the response
                is_match = "yes" in response_lower.split()[0:3]
                explanation = comparison_response
        
        except Exception as e:
            explanation = f"Error comparing: {str(e)}"
    
    # Save individual results
    results_filename = output_dir / f"{traj_name}_caption_results.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Trajectory Captioning Results\n")
        f.write(f"============================\n")
        f.write(f"Trajectory: {traj_name}\n")
        f.write(f"File path: {file_path}\n")
        f.write(f"\nAll Available Ground Truth Keys:\n")
        if possible_keys:
            for key_info in possible_keys:
                f.write(f"  - {key_info}\n")
        else:
            f.write("  No language instructions found in metadata\n")
        f.write(f"\nSelected Ground Truth Description:\n{ground_truth}\n")
        f.write(f"\nVLM Generated Caption:\n{vlm_caption}\n")
        f.write(f"\nSemantic Comparison:\n")
        f.write(f"Match: {'YES' if is_match else 'NO'}\n")
        f.write(f"Explanation: {explanation}\n")
        f.write(f"\nInput image saved as: {traj_name}_caption_input.jpg\n")
    
    return {
        "trajectory_name": traj_name,
        "ground_truth_description": ground_truth,
        "possible_ground_truth_keys": possible_keys,
        "vlm_caption": vlm_caption,
        "has_ground_truth": bool(ground_truth),
        "has_caption": bool(vlm_caption),
        "is_match": is_match,
        "comparison_explanation": explanation
    }


class TrajectoryCaptoningBenchmark:
    """Benchmark for evaluating trajectory captioning accuracy."""
    
    def __init__(self, dataset_path: str, output_dir: str = "./trajectory_captioning_results"):
        """
        Initialize the captioning benchmark.
        
        Args:
            dataset_path: Path to the directory containing VLA trajectory files or pattern
            output_dir: Directory to save captioning results
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure dataset for loading
        self.config = DatasetConfig(
            batch_size=4,
            shuffle=False,
            use_metadata=True,
            auto_build_metadata=False
        )
        
    def load_dataset(self, max_trajectories: Optional[int] = None) -> VLADataset:
        """
        Load the VLA dataset from the specified path.
        
        Args:
            max_trajectories: Maximum number of trajectories to process
            
        Returns:
            VLADataset ready for processing
        """
        print(f"Loading dataset from: {self.dataset_path}")
        
        # Create VLADataset
        dataset = VLADataset(
            path=self.dataset_path,
            return_type="numpy",
            config=self.config
        )
        
        total_trajectories = dataset.count()
        print(f"Found {total_trajectories} trajectory files")
        
        # Apply max_trajectories limit if specified
        if max_trajectories is not None and total_trajectories > max_trajectories:
            print(f"Limiting to {max_trajectories} trajectories")
            # Use take() to limit trajectories
            limited_items = dataset.take(max_trajectories)
            
            if limited_items:
                # Create limited dataset
                limited_file_paths = [item if isinstance(item, str) else item.get("item", str(item)) 
                                    for item in limited_items]
                
                import ray.data as rd
                limited_ray_dataset = rd.from_items(limited_file_paths)
                
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
        
        return dataset
    
    def run_benchmark(self, max_trajectories: Optional[int] = None) -> float:
        """
        Run the captioning benchmark on the dataset.
        
        Args:
            max_trajectories: Maximum number of trajectories to process
            
        Returns:
            Captioning accuracy score
        """
        print("\n" + "=" * 60)
        print("TRAJECTORY CAPTIONING ACCURACY BENCHMARK")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset(max_trajectories)
        
        # Process trajectories using the standalone function with output_dir
        from functools import partial
        process_fn = partial(process_single_trajectory_for_captioning, output_dir=self.output_dir)
        results_dataset = dataset.map(process_fn).materialize()
        results = list(results_dataset.iter_rows())
        
        # Calculate accuracy
        correct_matches = 0
        valid_comparisons = 0
        skipped_trajectories = 0
        
        # Track ground truth key statistics
        key_usage = {
            "language_instruction": 0,
            "current_task": 0,
            "language_instruction_2": 0,
            "language_instruction_3": 0
        }
        trajectories_with_multiple_keys = 0
        
        print("\nDetailed Caption Comparison Results:")
        print("-" * 80)
        
        for result in results:
            if "Skipped" in result.get("comparison_explanation", ""):
                skipped_trajectories += 1
                continue
            
            if result["has_ground_truth"] and result["has_caption"]:
                valid_comparisons += 1
                
                if result["is_match"]:
                    correct_matches += 1
                
                status = "✅" if result["is_match"] else "❌"
                print(f"{status} {result['trajectory_name']}: {'MATCH' if result['is_match'] else 'NO MATCH'}")
                print(f"   Explanation: {result['comparison_explanation']}")
                print()
        
        # Calculate accuracy
        accuracy = correct_matches / valid_comparisons if valid_comparisons > 0 else 0
        
        print(f"\nOverall Captioning Metrics:")
        print(f"Total trajectories: {len(results)}")
        print(f"Successful trajectories processed: {valid_comparisons}")
        print(f"Failed trajectories skipped: {skipped_trajectories}")
        print(f"Correct matches: {correct_matches}")
        print(f"Incorrect matches: {valid_comparisons - correct_matches}")
        print(f"Accuracy: {accuracy:.3f} ({correct_matches}/{valid_comparisons})")
        
        # Save summary
        summary_filename = self.output_dir / "captioning_accuracy_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"Trajectory Captioning Accuracy Summary\n")
            f.write(f"=====================================\n")
            f.write(f"Dataset path: {self.dataset_path}\n")
            f.write(f"Total trajectories: {len(results)}\n")
            f.write(f"Successful trajectories processed: {valid_comparisons}\n")
            f.write(f"Failed trajectories skipped: {skipped_trajectories}\n")
            f.write(f"Correct matches: {correct_matches}\n")
            f.write(f"Incorrect matches: {valid_comparisons - correct_matches}\n")
            f.write(f"Accuracy: {accuracy:.3f} ({correct_matches}/{valid_comparisons})\n")
        
        print(f"\n✅ Results saved to {self.output_dir}/")
        
        return accuracy


def main():
    """Main function to run the captioning benchmark."""
    parser = argparse.ArgumentParser(description="Run trajectory captioning benchmark on DROID dataset")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./droid_combined_data",
        help="Path to the directory containing VLA trajectory files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trajectory_captioning_results",
        help="Directory to save captioning results"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=400,
        help="Maximum number of trajectories to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create and run benchmark
        benchmark = TrajectoryCaptoningBenchmark(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        accuracy = benchmark.run_benchmark(max_trajectories=args.max_trajectories)
        
        print(f"\nFinal Trajectory Captioning Accuracy: {accuracy:.3f}")
        
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()