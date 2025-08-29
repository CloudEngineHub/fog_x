#!/usr/bin/env python3
"""
Simplified VLM Processing Example

This example provides a simple interface for processing robot trajectories with VLM:
- Input: List of DROID directories or MP4 files, and a question
- Output: Dictionary mapping input paths to VLM responses
- Uses parallel processing via Ray for efficiency
- Focuses only on perception data from MP4 videos

Usage:
    python simple_vlm_processing.py --trajectories /path/to/droid_dir1 /path/to/video2.mp4 \
        --question "Is this trajectory successful?"
"""

import argparse
import json
import os
import ray
import time
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from robodm.agent.tools import ToolsManager


def extract_frames_from_mp4(mp4_path: str, max_frames: int = 10) -> List[np.ndarray]:
    """
    Extract frames from an MP4 video file.
    
    Args:
        mp4_path: Path to the MP4 video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"    ⚠️ Could not open video file: {mp4_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            print(f"    ⚠️ No frames found in video: {mp4_path}")
            cap.release()
            return frames
            
        # Select frames evenly distributed throughout the video
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        print(f"    📹 Extracting {len(frame_indices)} frames from {total_frames} total frames (FPS: {fps:.1f})")
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                print(f"    ⚠️ Could not read frame {frame_idx}")
        
        cap.release()
        print(f"    ✅ Successfully extracted {len(frames)} frames from {os.path.basename(mp4_path)}")
        
    except Exception as e:
        print(f"    ❌ Error extracting frames from {mp4_path}: {e}")
        if 'cap' in locals():
            cap.release()
    
    return frames


def make_image_grid(images: List[np.ndarray], grid_cols: Optional[int] = None, target_size: Optional[tuple] = None) -> np.ndarray:
    """
    Create a tiled grid image from a list of RGB images.
    Images are resized to a common size and arranged row-wise.
    """
    if not images:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Determine grid columns
    num_images = len(images)
    if grid_cols is None or grid_cols <= 0:
        grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))

    # Determine target size
    if target_size is None:
        # Use median size to reduce distortion
        heights = [img.shape[0] for img in images if len(img.shape) == 3]
        widths = [img.shape[1] for img in images if len(img.shape) == 3]
        h = int(np.median(heights)) if heights else 480
        w = int(np.median(widths)) if widths else 640
        target_size = (w, h)

    # Resize all images
    resized = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        resized.append(cv2.resize(img, target_size))

    # Create grid canvas
    grid_h = target_size[1] * grid_rows
    grid_w = target_size[0] * grid_cols
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Paste images
    for idx, img in enumerate(resized):
        r = idx // grid_cols
        c = idx % grid_cols
        y0 = r * target_size[1]
        x0 = c * target_size[0]
        canvas[y0:y0 + target_size[1], x0:x0 + target_size[0], :] = img

    return canvas

def create_state_visualization(data: Dict[str, Any], max_frames: int = 10) -> List[np.ndarray]:
    # State visualization removed to focus purely on MP4 perception
    return []


def find_video_files_in_trajectory(trajectory_dir: str, video_path_key: str = None) -> List[str]:
    """
    Find MP4 video files in a DROID trajectory directory.
    
    Args:
        trajectory_dir: Path to DROID trajectory directory
        video_path_key: Specific video path key from metadata (e.g., 'ext1_mp4_path', 'wrist_mp4_path')
        
    Returns:
        List of paths to MP4 video files
    """
    video_files = []
    
    if video_path_key:
        # Use specific video path from metadata
        metadata_files = list(Path(trajectory_dir).glob("metadata_*.json"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            
            if video_path_key in metadata:
                # The metadata path is relative to GCS root, but we need local path
                relative_path = metadata[video_path_key]
                # Extract just the filename part
                video_filename = os.path.basename(relative_path)
                local_video_path = os.path.join(trajectory_dir, "recordings", "MP4", video_filename)
                
                if os.path.exists(local_video_path):
                    video_files = [local_video_path]
                    print(f"    📹 Using specified video: {video_path_key} -> {os.path.basename(local_video_path)}")
                else:
                    print(f"    ⚠️ Specified video {video_path_key} not found: {local_video_path}")
            else:
                print(f"    ⚠️ Video path key '{video_path_key}' not found in metadata")
    
    if not video_files:
        # Fallback to original logic - find all MP4 files
        # Try multiple potential directories
        potential_dirs = [
            os.path.join(trajectory_dir, "recordings", "MP4"),
            os.path.join(trajectory_dir, "recordings"),
            trajectory_dir
        ]
        
        for search_dir in potential_dirs:
            if os.path.exists(search_dir):
                mp4_pattern = os.path.join(search_dir, "*.mp4")
                found_files = glob.glob(mp4_pattern)
                
                # Filter out stereo files (we want the mono camera feeds)
                found_files = [f for f in found_files if '-stereo.mp4' not in f]
                
                if found_files:
                    video_files = found_files
                    print(f"    📁 Found {len(video_files)} video files in {search_dir}: {[os.path.basename(f) for f in video_files]}")
                    break
        
        if not video_files:
            print(f"    ⚠️ No video files found in any potential directory")
    
    return video_files


@ray.remote(num_cpus=1)
def process_single_trajectory(
    trajectory_path: str,
    question: str,
    tools_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    video_path_key: Optional[str] = None,
    num_frames: int = 6,
    passing_method: str = "stream",
    concat_grid_cols: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a single trajectory with VLM analysis.
    
    Args:
        trajectory_path: Path to a DROID directory or an MP4 file
        question: Question to ask the VLM
        tools_config: Configuration for VLM tools
        video_path_key: Specific video path key from metadata (for DROID directories only)
        
    Returns:
        Dictionary with trajectory analysis results
    """
    import os
    from pathlib import Path
    import cv2
    
    try:
        print(f"🔄 Processing {os.path.basename(trajectory_path)}")
        
        # Check if this is a DROID directory or trajectory file
        is_droid_directory = os.path.isdir(trajectory_path)
        images = []
        
        
        if is_droid_directory:
            # DROID directory format - extract frames from MP4 files
            print(f"  📁 Processing DROID directory: {os.path.basename(trajectory_path)}")
            
            # Find video files
            video_files = find_video_files_in_trajectory(trajectory_path, video_path_key)
            
            if video_files:
                # Use the first video file (typically exterior camera)
                primary_video = video_files[0]
                print(f"  📹 Using primary video: {os.path.basename(primary_video)}")
                
                # Extract frames from the video
                images = extract_frames_from_mp4(primary_video, max_frames=max(num_frames, 1))
                
                if not images:
                    print(f"  ⚠️ Failed to extract frames from video")
            else:
                print(f"  ⚠️ No video files found in DROID directory")
            
        else:
            # Direct MP4 file
            ext = os.path.splitext(trajectory_path.lower())[1]
            if ext == ".mp4":
                print(f"  🎞️ Processing MP4 file: {os.path.basename(trajectory_path)}")
                images = extract_frames_from_mp4(trajectory_path, max_frames=max(num_frames, 1))
            else:
                print(f"  ❌ Unsupported input (expected directory or .mp4): {trajectory_path}")
                images = []
        
        # Prepare images for VLM analysis
        if len(images) == 0:
            return {
                "trajectory_path": trajectory_path,
                "success": False,
                "error": "No images found in input",
                "vlm_response": None
            }
        
        # Select representative frames for analysis
        num_frames_to_use = min(max(num_frames, 1), len(images))
        if len(images) > num_frames_to_use:
            # Select frames evenly distributed throughout trajectory
            indices = np.linspace(0, len(images) - 1, num_frames_to_use, dtype=int)
            selected_images = [images[i] for i in indices]
        else:
            selected_images = list(images)
        
        # Prepare frames for VLM analysis
        processed_frames = []
        for img in selected_images:
            if len(img.shape) == 3:
                processed_frames.append(img)
            elif len(img.shape) == 2:
                processed_frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            else:
                processed_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        # Initialize VLM tools
        tools_manager = ToolsManager(config=tools_config)
        
        # Get the VLM tool
        vlm_tool = tools_manager.get_tool("robo2vlm")
        
        traj_name = os.path.splitext(os.path.basename(trajectory_path))[0]

        frame_responses = []
        if passing_method == "stream":
            # Pass all frames together with a single prompt (no per-frame captioning)
            final_prompt = f"""These are {len(processed_frames)} evenly sampled frames from a robot trajectory in temporal order. Considering them together, does the trajectory look successful? First answer yes or no, then explain why."""
            vlm_response = vlm_tool(processed_frames, final_prompt)
            processing_method_used = "all_frames_stream"
        else:
            # Concatenate frames into a tiled grid and analyze once
            grid_image = make_image_grid(processed_frames, grid_cols=concat_grid_cols)
            final_prompt = f"""This image is a tiled grid of {len(processed_frames)} evenly sampled frames from a robot trajectory (ordered left-to-right, top-to-bottom). Based on this sequence, does the trajectory look successful? First answer yes or no, then explain why."""
            vlm_response = vlm_tool(grid_image, final_prompt)
            processing_method_used = "concat_grid"
            # Optionally save the grid image
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                grid_path = Path(output_dir) / f"{traj_name}_grid.jpg"
                cv2.imwrite(str(grid_path), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        
        # Extract success prediction from VLM response (aligned with droid_vlm_demo.py)
        response_lower = vlm_response.lower()
        
        # Look for clear yes/no indicators in the response
        if "answer: **yes**" in response_lower or "answer: yes" in response_lower:
            vlm_prediction = True
        elif "answer: **no**" in response_lower or "answer: no" in response_lower:
            vlm_prediction = False
        else:
            # Fallback to simple yes/no check in first part of response
            first_part = ' '.join(response_lower.split()[:10])
            vlm_prediction = "yes" in first_part and "no" not in first_part
        
        print(f"  ✅ VLM Response: '{vlm_response[:100]}...'")
        print(f"  🎯 Success Prediction: {vlm_prediction}")
        
        # Save results to output directory if specified
        if output_dir:            
            os.makedirs(output_dir, exist_ok=True)
            results_dir = Path(output_dir)
            
            # Save individual frames for inspection (stream mode passes all frames together)
            if passing_method == "stream":
                for i, frame in enumerate(processed_frames):
                    frame_filename = results_dir / f"{traj_name}_frame_{i+1}.jpg"
                    cv2.imwrite(str(frame_filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Save detailed results
            results_filename = results_dir / f"{traj_name}_results.txt"
            with open(results_filename, 'w') as f:
                f.write(f"VLM Processing Results ({'Frame-by-Frame' if passing_method=='stream' else 'Concat Grid'})\n")
                f.write(f"======================================\n")
                f.write(f"Trajectory: {traj_name}\n")
                f.write(f"File path: {trajectory_path}\n")
                f.write(f"VLM prediction (success): {vlm_prediction}\n")
                f.write(f"Frames analyzed: {num_frames_to_use}/{len(images)}\n")
                if passing_method == 'stream':
                    f.write(f"\n--- Frames Provided ---\n")
                    f.write(f"{len(processed_frames)} frames were analyzed together in one request.\n")
                f.write(f"\n--- Final Analysis ---\n")
                f.write(f"Final Prompt:\n{final_prompt}\n")
                f.write(f"\nFinal VLM Response:\n{vlm_response}\n")
                if passing_method == 'stream':
                    f.write(f"\nFrames saved as: {traj_name}_frame_1.jpg to {traj_name}_frame_{len(processed_frames)}.jpg\n")
        
        return {
            "trajectory_path": trajectory_path,
            "success": True,
            "error": None,
            "vlm_response": vlm_response,
            "vlm_prediction": vlm_prediction,
            "frames_analyzed": num_frames_to_use,
            "total_frames": len(images),
            "frame_responses": frame_responses,
            "processing_method": processing_method_used,
            "passing_method": passing_method,
            "num_frames": num_frames_to_use
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {trajectory_path}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "trajectory_path": trajectory_path,
            "success": False,
            "error": str(e),
            "vlm_response": None
        }


def process_trajectories_parallel(
    trajectory_paths: List[str],
    question: str,
    max_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    video_path_key: Optional[str] = None,
    num_frames: Optional[int] = None,
    passing_method: str = "stream",
    concat_grid_cols: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple trajectories in parallel with VLM analysis.
    
    Args:
        trajectory_paths: List of DROID directories or MP4 files
        question: Question to ask the VLM (e.g., "Is this trajectory successful?")
        max_workers: Maximum number of parallel workers (None for automatic)
        video_path_key: Specific video path key from metadata (for DROID directories only)
        
    Returns:
        Dictionary mapping trajectory paths to analysis results
    """
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Configure VLM tools
    tools_config = {
        "tools": {
            "robo2vlm": {
                "model": "Qwen/Qwen2.5-VL-32B-Instruct",
                "temperature": 0.1,
                "max_tokens": 4096,
                "context_length": 1024
            }
        }
    }
    
    print(f"🚀 Starting parallel processing of {len(trajectory_paths)} trajectories")
    print(f"📊 Configuration:")
    print(f"   Question: {question}")
    if num_frames is not None:
        print(f"   Num frames: {num_frames}")
    print(f"   Passing method: {passing_method}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {output_dir}")
    
    # Submit all tasks to Ray
    futures = []
    for traj_path in trajectory_paths:
        future = process_single_trajectory.remote(
            trajectory_path=traj_path,
            question=question,
            tools_config=tools_config,
            output_dir=output_dir,
            video_path_key=video_path_key,
            num_frames=(num_frames if num_frames is not None else 6),
            passing_method=passing_method,
            concat_grid_cols=concat_grid_cols
        )
        futures.append(future)
    
    # Collect results as they complete
    results = {}
    completed = 0
    start_time = time.time()
    
    while futures:
        # Wait for at least one task to complete
        ready, futures = ray.wait(futures, num_returns=1, timeout=30.0)
        
        for future in ready:
            result = ray.get(future)
            completed += 1
            
            traj_path = result["trajectory_path"]
            results[traj_path] = result
            
            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(trajectory_paths) - completed) / rate if rate > 0 else 0
            
            status = "✅" if result["success"] else "❌"
            print(f"{status} [{completed}/{len(trajectory_paths)}] {os.path.basename(traj_path)} "
                  f"(Rate: {rate:.1f}/min, ETA: {eta/60:.1f}min)")
    
    total_time = time.time() - start_time
    successful_processing = sum(1 for r in results.values() if r["success"])
    failed_processing = len(results) - successful_processing
    
    # Count VLM predictions
    vlm_success_predictions = sum(1 for r in results.values() if r["success"] and r.get("vlm_prediction", False))
    vlm_failure_predictions = sum(1 for r in results.values() if r["success"] and not r.get("vlm_prediction", False))
    
    print(f"\n📈 Processing Complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successfully processed: {successful_processing}")
    print(f"   Failed to process: {failed_processing}")
    print(f"   VLM Success predictions: {vlm_success_predictions}")
    print(f"   VLM Failure predictions: {vlm_failure_predictions}")
    print(f"   Rate: {len(trajectory_paths)/total_time*60:.1f} trajectories/minute")
    
    # Save summary if output directory is specified
    if output_dir:
        summary_file = os.path.join(output_dir, "processing_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"VLM Processing Summary\n")
            f.write(f"====================\n")
            f.write(f"Total trajectories: {len(trajectory_paths)}\n")
            f.write(f"Successfully processed: {successful_processing}\n")
            f.write(f"Failed to process: {failed_processing}\n")
            f.write(f"VLM Success predictions: {vlm_success_predictions}\n")
            f.write(f"VLM Failure predictions: {vlm_failure_predictions}\n")
            f.write(f"Processing time: {total_time:.1f}s\n")
            f.write(f"Processing rate: {len(trajectory_paths)/total_time*60:.1f} trajectories/minute\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Question: {question}\n")
        print(f"📄 Summary saved to {summary_file}")
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Simplified VLM Processing for Robot Trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with DROID directories or MP4s
    python simple_vlm_processing.py \
        --trajectories /path/to/droid_dir1 /path/to/video2.mp4 \
        --question "Is this trajectory successful?"
        """)
    
    parser.add_argument(
        "--trajectories", 
        nargs="+", 
        required=True,
        help="Paths to DROID directories or MP4 files"
    )
    parser.add_argument(
        "--question", 
        required=True,
        help="Question to ask the VLM (e.g., 'Is this trajectory successful?')"
    )
    parser.add_argument(
        "--output", 
        help="Output file path for results (JSON format). If not specified, prints to stdout"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory for saving detailed results (prompt, input images, VLM responses)"
    )
    parser.add_argument(
        "--video-path-key",
        help="Specific video path key from metadata (e.g., 'ext1_mp4_path', 'wrist_mp4_path')"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Number of evenly sampled frames to use (default: 6)"
    )
    parser.add_argument(
        "--passing-method",
        choices=["stream", "concat"],
        default="stream",
        help="How to pass images to VLM: per-frame ('stream') or tiled grid ('concat')"
    )
    parser.add_argument(
        "--concat-grid-cols",
        type=int,
        help="Number of columns for concatenated grid (concat mode). Default sqrt(N)."
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns and validate paths
    trajectory_paths = []
    for path_pattern in args.trajectories:
        if "*" in path_pattern:
            # Handle glob patterns
            from glob import glob
            matched_paths = glob(path_pattern)
            trajectory_paths.extend(matched_paths)
        else:
            trajectory_paths.append(path_pattern)
    
    # Filter for valid inputs and check existence (directories or .mp4)
    valid_paths = []
    for path in trajectory_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                valid_paths.append(path)
            else:
                ext = os.path.splitext(path.lower())[1]
                if ext == ".mp4":
                    valid_paths.append(path)
                else:
                    print(f"⚠️  Skipping {path}: unsupported format (expected directory or .mp4)")
        else:
            print(f"⚠️  Skipping {path}: file does not exist")
    
    if not valid_paths:
        print("❌ No valid inputs found (directories or .mp4)!")
        return 1
    
    print(f"📂 Found {len(valid_paths)} valid inputs")
    
    # Process trajectories
    try:
        results = process_trajectories_parallel(
            trajectory_paths=valid_paths,
            question=args.question,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            video_path_key=args.video_path_key,
            num_frames=args.num_frames,
            passing_method=args.passing_method,
            concat_grid_cols=args.concat_grid_cols
        )
        
        # Output results
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Results saved to {args.output}")
        else:
            print("\n📋 Results:")
            print("=" * 60)
            for path, result in results.items():
                print(f"\n🗂️  {os.path.basename(path)}:")
                if result["success"]:
                    print(f"   🎯 VLM Prediction: {'Success' if result.get('vlm_prediction', False) else 'Failure'}")
                    print(f"   🤖 VLM Response: {result['vlm_response'][:200]}...")
                    print(f"   📊 Frames: {result.get('frames_analyzed', 0)}/{result.get('total_frames', 0)}")
                else:
                    print(f"   ❌ Error: {result['error']}")
        
        # Print output directory info if used
        if args.output_dir:
            print(f"\n📁 Detailed results saved to: {args.output_dir}/")
            print(f"   - Individual result files: *_results.txt")
            print(f"   - Individual frame images: *_frame_N.jpg")
            print(f"   - Processing summary: processing_summary.txt")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    exit(main())