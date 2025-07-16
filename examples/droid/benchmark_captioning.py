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
    
    # First, check if we have metadata and if it contains raw_data_path
    current_task = None
    if 'metadata' in trajectory:
        metadata = trajectory['metadata']
        if hasattr(metadata, '__len__') and len(metadata) > 0:
            metadata_val = metadata[0]
            if isinstance(metadata_val, str):
                try:
                    import json
                    decoded_metadata = json.loads(metadata_val)
                    raw_data_path = decoded_metadata.get('raw_data_path', '')
                    
                    # Try to load the raw metadata JSON file to get current_task
                    if raw_data_path:
                        # Construct metadata JSON path from raw_data_path
                        import os
                        import glob
                        metadata_pattern = os.path.join(raw_data_path, 'metadata_*.json')
                        metadata_files = glob.glob(metadata_pattern)
                        
                        if metadata_files:
                            with open(metadata_files[0], 'r') as f:
                                raw_metadata = json.load(f)
                                current_task = raw_metadata.get('current_task', '')
                                if current_task:
                                    possible_keys.append(f"raw_metadata/current_task: {current_task}")
                                    key_candidates.append("raw_metadata/current_task")
                                    trajectory["raw_metadata/current_task"] = current_task
                except Exception as e:
                    print(f"Error loading raw metadata: {e}")
    
    try:        
        # Look for language instruction keys directly in the trajectory
        found_instructions = []
        
        for key in key_candidates:
            if key == "raw_metadata/current_task":
                # We already have current_task from above
                if current_task:
                    found_instructions.append(current_task)
            else:
                value = trajectory.get(key, "")
                
                # Check if value exists and has content
                has_content = False
                value_str = ""
                
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        # Handle byte strings
                        val = value[0]
                        if isinstance(val, bytes):
                            value_str = val.decode('utf-8')
                        else:
                            value_str = str(val)
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
    
    # Skip if no language instructions found
    if not ground_truth:
        print(f"⚠️  Skipping {traj_name} - no language instructions found")
        return {"results": [{
            "trajectory_name": traj_name,
            "camera_view": "none",
            "ground_truth_description": "",
            "possible_ground_truth_keys": possible_keys,
            "vlm_caption": "",
            "has_ground_truth": False,
            "has_caption": False,
            "is_match": False,
            "comparison_explanation": "Skipped - no language instructions"
        }]}
    
    # Process both exterior cameras
    results_per_camera = []
    
    # Find camera keys
    camera_keys = []
    exterior_cameras = {}
    
    for key in trajectory.keys():
        if "raw/images/" in key or "observation/images/" in key or "image" in key.lower():
            camera_keys.append(key)
            # Check for exterior cameras
            if "exterior" in key or "ext" in key:
                if "1" in key or "image_1" in key:
                    exterior_cameras["exterior_1"] = key
                elif "2" in key or "image_2" in key:
                    exterior_cameras["exterior_2"] = key
    
    # If no exterior cameras found, skip
    if not exterior_cameras:
        print(f"⚠️  Skipping {traj_name} - no exterior cameras found")
        return {"results": [{
            "trajectory_name": traj_name,
            "camera_view": "none",
            "ground_truth_description": ground_truth,
            "possible_ground_truth_keys": possible_keys,
            "vlm_caption": "",
            "has_ground_truth": True,
            "has_caption": False,
            "is_match": False,
            "comparison_explanation": "No exterior cameras found"
        }]}
    
    # Process each exterior camera
    for camera_name, camera_key in exterior_cameras.items():
        vlm_caption = ""
        is_match = False
        explanation = ""
        
        try:
            # Initialize VLM service locally
            vlm_service = get_vlm_service()
            vlm_service.initialize()
            
            frames = trajectory.get(camera_key, [])
            
            if len(frames) >= 6:
                # Extract 6 frames evenly distributed
                num_frames = 6
                indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
                selected_frames = [frames[i] for i in indices]
                
                # Create 2x3 grid
                top_row = np.hstack(selected_frames[:3])
                bottom_row = np.hstack(selected_frames[3:])
                stitched_frame = np.vstack([top_row, bottom_row])
                
                # Ensure image is uint8 before saving
                if stitched_frame.dtype != np.uint8:
                    # Check if values are in [0, 1] range (common for float images)
                    if stitched_frame.dtype in [np.float32, np.float64] and stitched_frame.max() <= 1.0:
                        # Convert from [0, 1] to [0, 255]
                        stitched_frame = (stitched_frame * 255).astype(np.uint8)
                    else:
                        # Already in [0, 255] range, just convert type
                        stitched_frame = np.clip(stitched_frame, 0, 255).astype(np.uint8)
                
                # Save input image with camera name
                image_filename = output_dir / f"{traj_name}_{camera_name}_caption_input.jpg"
                cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                
                # Generate caption
                vlm_prompt = (
                    "These are 6 frames from a robot trajectory shown in temporal order "
                    "(left to right, top to bottom). Please describe with one sentence what task the robot "
                    "is performing in this trajectory. Be very specific about the "
                    "actions and objects involved."
                )
                
                vlm_caption = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                print(f"  {camera_name}: Generated caption")
        
        except Exception as e:
            print(f"Error generating caption for {traj_name} {camera_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Compare descriptions
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
        
        # Save individual results for this camera
        results_filename = output_dir / f"{traj_name}_{camera_name}_caption_results.txt"
        with open(results_filename, 'w') as f:
            f.write(f"Trajectory Captioning Results - {camera_name}\n")
            f.write(f"=========================================\n")
            f.write(f"Trajectory: {traj_name}\n")
            f.write(f"Camera View: {camera_name}\n")
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
            f.write(f"\nInput image saved as: {traj_name}_{camera_name}_caption_input.jpg\n")
        
        # Add result for this camera
        results_per_camera.append({
            "trajectory_name": traj_name,
            "camera_view": camera_name,
            "ground_truth_description": ground_truth,
            "possible_ground_truth_keys": possible_keys,
            "vlm_caption": vlm_caption,
            "has_ground_truth": bool(ground_truth),
            "has_caption": bool(vlm_caption),
            "is_match": is_match,
            "comparison_explanation": explanation
        })
    
    # Wrap in dict for Ray compatibility
    return {"results": results_per_camera}


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
        results_lists = list(results_dataset.iter_rows())
        
        # Flatten results since each trajectory returns a dict with list of results (one per camera)
        results = []
        for result_dict in results_lists:
            if isinstance(result_dict, dict) and "results" in result_dict:
                results.extend(result_dict["results"])
            elif isinstance(result_dict, list):
                # Handle old format for backward compatibility
                results.extend(result_dict)
            else:
                # Single result dict
                results.append(result_dict)
        
        # Calculate accuracy per camera view
        camera_stats = {}
        overall_correct_matches = 0
        overall_valid_comparisons = 0
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
            camera_view = result.get("camera_view", "unknown")
            
            # Initialize camera stats if needed
            if camera_view not in camera_stats:
                camera_stats[camera_view] = {
                    "correct_matches": 0,
                    "valid_comparisons": 0,
                    "skipped": 0
                }
            
            if "Skipped" in result.get("comparison_explanation", ""):
                skipped_trajectories += 1
                camera_stats[camera_view]["skipped"] += 1
                continue
            
            if result["has_ground_truth"] and result["has_caption"]:
                camera_stats[camera_view]["valid_comparisons"] += 1
                overall_valid_comparisons += 1
                
                if result["is_match"]:
                    camera_stats[camera_view]["correct_matches"] += 1
                    overall_correct_matches += 1
                
                status = "✅" if result["is_match"] else "❌"
                print(f"{status} {result['trajectory_name']} ({camera_view}): {'MATCH' if result['is_match'] else 'NO MATCH'}")
                print(f"   Explanation: {result['comparison_explanation']}")
                print()
        
        # Calculate overall accuracy
        overall_accuracy = overall_correct_matches / overall_valid_comparisons if overall_valid_comparisons > 0 else 0
        
        print(f"\nOverall Captioning Metrics:")
        print(f"Total trajectory-camera pairs: {len(results)}")
        print(f"Successful comparisons: {overall_valid_comparisons}")
        print(f"Failed/skipped: {skipped_trajectories}")
        print(f"Correct matches: {overall_correct_matches}")
        print(f"Incorrect matches: {overall_valid_comparisons - overall_correct_matches}")
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_correct_matches}/{overall_valid_comparisons})")
        
        # Print per-camera statistics
        print(f"\nPer-Camera View Statistics:")
        print("-" * 50)
        for camera_view, stats in sorted(camera_stats.items()):
            if stats["valid_comparisons"] > 0:
                camera_accuracy = stats["correct_matches"] / stats["valid_comparisons"]
                print(f"{camera_view}:")
                print(f"  Valid comparisons: {stats['valid_comparisons']}")
                print(f"  Correct matches: {stats['correct_matches']}")
                print(f"  Accuracy: {camera_accuracy:.3f} ({stats['correct_matches']}/{stats['valid_comparisons']})")
                print(f"  Skipped: {stats['skipped']}")
        
        # Save summary
        summary_filename = self.output_dir / "captioning_accuracy_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"Trajectory Captioning Accuracy Summary\n")
            f.write(f"=====================================\n")
            f.write(f"Dataset path: {self.dataset_path}\n")
            f.write(f"Total trajectory-camera pairs: {len(results)}\n")
            f.write(f"Successful comparisons: {overall_valid_comparisons}\n")
            f.write(f"Failed/skipped: {skipped_trajectories}\n")
            f.write(f"Correct matches: {overall_correct_matches}\n")
            f.write(f"Incorrect matches: {overall_valid_comparisons - overall_correct_matches}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_correct_matches}/{overall_valid_comparisons})\n")
            f.write(f"\nPer-Camera View Statistics:\n")
            f.write("-" * 50 + "\n")
            for camera_view, stats in sorted(camera_stats.items()):
                if stats["valid_comparisons"] > 0:
                    camera_accuracy = stats["correct_matches"] / stats["valid_comparisons"]
                    f.write(f"{camera_view}:\n")
                    f.write(f"  Valid comparisons: {stats['valid_comparisons']}\n")
                    f.write(f"  Correct matches: {stats['correct_matches']}\n")
                    f.write(f"  Accuracy: {camera_accuracy:.3f} ({stats['correct_matches']}/{stats['valid_comparisons']})\n")
                    f.write(f"  Skipped: {stats['skipped']}\n")
        
        print(f"\n✅ Results saved to {self.output_dir}/")
        
        return overall_accuracy


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