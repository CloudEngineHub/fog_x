#!/usr/bin/env python3
"""
Simplified VLM Processing Example

This example provides a simple interface for processing robot trajectories with VLM:
- Input: List of trajectory paths, image key, language key, question
- Output: Dictionary mapping trajectory paths to VLM responses
- Uses parallel processing via Ray for efficiency
- Works with both HDF5 and VLA trajectory formats

Usage:
    python simple_vlm_processing.py --trajectories path1.h5 path2.h5 path3.vla \
        --image-key "observation/images/hand_camera" \
        --language-key "metadata/language_instruction" \
        --question "Is this trajectory successful?"
"""

import argparse
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

from robodm import Trajectory
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


def find_video_files_in_trajectory(trajectory_dir: str) -> List[str]:
    """
    Find MP4 video files in a DROID trajectory directory.
    
    Args:
        trajectory_dir: Path to DROID trajectory directory
        
    Returns:
        List of paths to MP4 video files
    """
    # Look for MP4 files in recordings/MP4/ directory
    mp4_pattern = os.path.join(trajectory_dir, "recordings", "MP4", "*.mp4")
    video_files = glob.glob(mp4_pattern)
    
    # Filter out stereo files (we want the mono camera feeds)
    mono_files = [f for f in video_files if '-stereo.mp4' not in f]
    
    print(f"    📁 Found {len(mono_files)} video files: {[os.path.basename(f) for f in mono_files]}")
    
    return mono_files


def create_state_visualization(data: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Create visualizations from robot state data when no images are available.
    
    Args:
        data: Dictionary containing trajectory data
        
    Returns:
        List of visualization images as numpy arrays
    """
    visualizations = []
    
    # Get key state data
    actions = data.get('action', None)
    joint_positions = data.get('observation/state/joint_positions', None)
    cartesian_position = data.get('observation/state/cartesian_position', None)
    gripper_position = data.get('observation/state/gripper_position', None)
    
    if actions is None:
        # No action data available
        return [np.zeros((224, 224, 3), dtype=np.uint8)]
    
    num_timesteps = len(actions)
    time_steps = np.arange(num_timesteps)
    
    # Create 4 different visualizations
    fig_size = (6, 4)
    
    # 1. Action trajectory over time
    plt.figure(figsize=fig_size)
    plt.title('Robot Actions Over Time')
    for i in range(min(actions.shape[1], 6)):  # Plot up to 6 action dimensions
        plt.plot(time_steps, actions[:, i], label=f'Action {i}', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Action Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to numpy array
    plt.savefig('/tmp/action_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    img = cv2.imread('/tmp/action_plot.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    visualizations.append(img)
    
    # 2. Joint positions (if available)
    if joint_positions is not None:
        plt.figure(figsize=fig_size)
        plt.title('Joint Positions Over Time')
        for i in range(min(joint_positions.shape[1], 7)):
            plt.plot(time_steps, joint_positions[:, i], label=f'Joint {i}', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Joint Position (rad)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('/tmp/joint_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        img = cv2.imread('/tmp/joint_plot.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        visualizations.append(img)
    
    # 3. Cartesian position trajectory (if available)
    if cartesian_position is not None:
        plt.figure(figsize=fig_size)
        plt.title('Cartesian Position Trajectory')
        
        # Plot 3D trajectory
        if cartesian_position.shape[1] >= 3:
            # Position trajectory
            plt.subplot(2, 1, 1)
            plt.plot(time_steps, cartesian_position[:, 0], label='X', alpha=0.8)
            plt.plot(time_steps, cartesian_position[:, 1], label='Y', alpha=0.8)
            plt.plot(time_steps, cartesian_position[:, 2], label='Z', alpha=0.8)
            plt.ylabel('Position (m)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Orientation (if available)
            if cartesian_position.shape[1] >= 6:
                plt.subplot(2, 1, 2)
                plt.plot(time_steps, cartesian_position[:, 3], label='Roll', alpha=0.8)
                plt.plot(time_steps, cartesian_position[:, 4], label='Pitch', alpha=0.8)
                plt.plot(time_steps, cartesian_position[:, 5], label='Yaw', alpha=0.8)
                plt.ylabel('Orientation (rad)')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.xlabel('Time Step')
        plt.tight_layout()
        
        plt.savefig('/tmp/cartesian_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        img = cv2.imread('/tmp/cartesian_plot.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        visualizations.append(img)
    
    # 4. Gripper position (if available)
    if gripper_position is not None:
        plt.figure(figsize=fig_size)
        plt.title('Gripper Position Over Time')
        plt.plot(time_steps, gripper_position, 'b-', linewidth=2, label='Gripper Position')
        plt.xlabel('Time Step')
        plt.ylabel('Gripper Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add horizontal lines for typical open/closed positions
        plt.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Closed')
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Open')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('/tmp/gripper_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        img = cv2.imread('/tmp/gripper_plot.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        visualizations.append(img)
    
    # Ensure we have at least 4 visualizations by padding with the action plot
    while len(visualizations) < 4:
        visualizations.append(visualizations[0])
    
    return visualizations


@ray.remote(num_cpus=1)
def process_single_trajectory(
    trajectory_path: str,
    image_key: str,
    language_key: str,
    question: str,
    tools_config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single trajectory with VLM analysis.
    
    Args:
        trajectory_path: Path to the trajectory file (.h5) or directory (DROID format)
        image_key: Key to extract image data from trajectory (ignored for DROID MP4 processing)
        language_key: Key to extract language instruction from trajectory  
        question: Question to ask the VLM
        tools_config: Configuration for VLM tools
        
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
        language_instruction = None
        use_state_visualization = False
        
        if is_droid_directory:
            # DROID directory format - extract frames from MP4 files
            print(f"  📁 Processing DROID directory: {os.path.basename(trajectory_path)}")
            
            # Find video files
            video_files = find_video_files_in_trajectory(trajectory_path)
            
            if video_files:
                # Use the first video file (typically exterior camera)
                primary_video = video_files[0]
                print(f"  📹 Using primary video: {os.path.basename(primary_video)}")
                
                # Extract frames from the video
                images = extract_frames_from_mp4(primary_video, max_frames=10)
                
                if not images:
                    print(f"  ⚠️ Failed to extract frames from video, falling back to state visualization")
                    use_state_visualization = True
            else:
                print(f"  ⚠️ No video files found in DROID directory")
                use_state_visualization = True
            
            # Try to extract language instruction from HDF5 file
            hdf5_file = os.path.join(trajectory_path, "trajectory.h5")
            if os.path.exists(hdf5_file):
                try:
                    traj = Trajectory(hdf5_file, mode="r")
                    data = traj.load()
                    traj.close()
                    
                    if language_key in data:
                        lang_data = data[language_key]
                        if isinstance(lang_data, np.ndarray):
                            if lang_data.ndim == 0:
                                language_instruction = str(lang_data.item())
                            else:
                                language_instruction = str(lang_data[0])
                        else:
                            language_instruction = str(lang_data)
                        
                        # Handle byte strings
                        if isinstance(language_instruction, str) and language_instruction.startswith("b'"):
                            language_instruction = language_instruction[2:-1]
                        
                        print(f"  📝 Language instruction: '{language_instruction[:50]}...'")
                    else:
                        print(f"  ⚠️ Language key '{language_key}' not found in HDF5 file")
                        
                    # Fall back to state visualization if no images from video
                    if use_state_visualization:
                        print(f"  📊 Creating state-based visualization from HDF5 data")
                        images = create_state_visualization(data)
                        
                except Exception as e:
                    print(f"  ⚠️ Could not load language instruction from HDF5: {e}")
            
        else:
            # Traditional trajectory file format
            traj = Trajectory(trajectory_path, mode="r")
            try:
                data = traj.load()
            except Exception as e:
                print(f"  ❌ Error loading trajectory data: {e}")
                print(f"  📋 Attempting to load individual streams...")
                
                # Try to load streams individually to identify problematic ones
                streams = traj.backend.get_streams()
                data = {}
                problematic_streams = []
                
                for stream in streams:
                    try:
                        stream_data = traj.backend.read_feature_data(stream.feature_name)
                        if stream_data is not None:
                            data[stream.feature_name] = stream_data
                            print(f"    ✅ Loaded {stream.feature_name}: {stream_data.shape}")
                        else:
                            print(f"    ⚠️ No data for {stream.feature_name}")
                    except Exception as stream_e:
                        print(f"    ❌ Failed to load {stream.feature_name}: {stream_e}")
                        problematic_streams.append(stream.feature_name)
                
                if problematic_streams:
                    print(f"  📋 Skipping problematic streams: {problematic_streams}")
            
            traj.close()
            
            # Extract image data or create visualizations from state data
            if image_key in data:
                images = data[image_key]
                print(f"  📷 Found {len(images)} images with shape {images[0].shape if len(images) > 0 else 'None'}")
            else:
                available_image_keys = [k for k in data.keys() if 'image' in k.lower()]
                if available_image_keys:
                    print(f"  ⚠️ Image key '{image_key}' not found, but found: {available_image_keys}")
                    # Use the first available image key
                    image_key = available_image_keys[0]
                    images = data[image_key]
                    print(f"  📷 Using {image_key} with {len(images)} images")
                else:
                    # No images available - create state visualization
                    print(f"  📊 No images found, creating state-based visualization")
                    use_state_visualization = True
                    images = create_state_visualization(data)
            
            # Extract language instruction
            if language_key in data:
                lang_data = data[language_key]
                if isinstance(lang_data, np.ndarray):
                    if lang_data.ndim == 0:
                        # Scalar
                        language_instruction = str(lang_data.item())
                    else:
                        # Array - take first element
                        language_instruction = str(lang_data[0])
                else:
                    language_instruction = str(lang_data)
                
                # Handle byte strings
                if isinstance(language_instruction, str) and language_instruction.startswith("b'"):
                    language_instruction = language_instruction[2:-1]  # Remove b' and '
                
                print(f"  📝 Language instruction: '{language_instruction[:50]}...'")
            else:
                available_keys = [k for k in data.keys() if 'language' in k.lower() or 'instruction' in k.lower()]
                print(f"  ⚠️  Language key '{language_key}' not found. Available keys: {available_keys}")
        
        # Prepare images for VLM analysis
        if len(images) == 0:
            return {
                "trajectory_path": trajectory_path,
                "success": False,
                "error": "No images found in trajectory",
                "vlm_response": None,
                "language_instruction": language_instruction
            }
        
        # Select representative frames for analysis
        num_frames_to_use = min(6, len(images))
        if len(images) > num_frames_to_use:
            # Select frames evenly distributed throughout trajectory
            indices = np.linspace(0, len(images) - 1, num_frames_to_use, dtype=int)
            selected_images = [images[i] for i in indices]
        else:
            selected_images = list(images)
        
        # Create image grid for VLM analysis
        if num_frames_to_use <= 4:
            # Create 2x2 grid
            rows = 2
            cols = 2
            # Pad with copies if needed
            while len(selected_images) < 4:
                selected_images.append(selected_images[-1])
        else:
            # Create 2x3 grid
            rows = 2
            cols = 3
            # Pad with copies if needed
            while len(selected_images) < 6:
                selected_images.append(selected_images[-1])
        
        # Resize images to consistent size for grid
        target_height, target_width = 224, 224
        resized_images = []
        for img in selected_images:
            if len(img.shape) == 3:  # RGB image
                resized = cv2.resize(img, (target_width, target_height))
                resized_images.append(resized)
            else:
                # Handle grayscale or other formats
                resized_images.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))
        
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_images = resized_images[r * cols:(r + 1) * cols]
            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)
        
        grid_image = np.vstack(grid_rows)
        
        # Initialize VLM tools
        tools_manager = ToolsManager(config=tools_config)
        
        # Get the VLM tool
        vlm_tool = tools_manager.get_tool("robo2vlm")
        
        # Prepare VLM prompt aligned with droid_vlm_demo.py
        context = f"\nLanguage instruction: '{language_instruction}'" if language_instruction else ""
        traj_name = os.path.splitext(os.path.basename(trajectory_path))[0]
        
        if use_state_visualization:
            # Enhanced prompt for state visualization similar to droid_vlm_demo.py
            full_prompt = f"""These are {num_frames_to_use} robot state visualizations from a trajectory. Does this trajectory look successful? First answer yes or no, then explain why.

The plots show:
1. Robot actions over time (control commands)
2. Joint positions over time (robot arm configuration)  
3. Cartesian position trajectory (end-effector path)
4. Gripper position over time (open/close state)

CRITICAL: Smooth-looking trajectories do NOT always mean success! Many robot failures appear smooth but fail to achieve the task goal.

For success classification, look for:
- SUCCESSFUL: Goal achievement indicators - reaching target positions, completing full task sequence, appropriate final states
- FAILED: Task incompletion signs - stopping short of targets, incomplete motion sequences, premature endings, suboptimal final positions

Key failure patterns to identify:
- Trajectories that end prematurely or don't reach intended targets
- Motion that looks controlled but accomplishes nothing meaningful  
- Missing expected motion phases (approach, grasp, transport, place)
- Final gripper/joint positions that suggest incomplete tasks

First answer yes or no, then explain your reasoning based on task completion evidence.{context}"""
        else:
            # Align with droid_vlm_demo.py pattern for image analysis
            full_prompt = f"""These are {num_frames_to_use} frames from a robot trajectory. Does this trajectory look successful? First answer yes or no, then explain why.{context}"""
        
        # Call VLM
        vlm_response = vlm_tool(grid_image, full_prompt)
        
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
            
            # Save input image
            image_filename = results_dir / f"{traj_name}_input.jpg"
            cv2.imwrite(str(image_filename), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            
            # Save detailed results
            results_filename = results_dir / f"{traj_name}_results.txt"
            with open(results_filename, 'w') as f:
                f.write(f"VLM Processing Results\n")
                f.write(f"===================\n")
                f.write(f"Trajectory: {traj_name}\n")
                f.write(f"File path: {trajectory_path}\n")
                f.write(f"VLM prediction (success): {vlm_prediction}\n")
                f.write(f"Language instruction: {language_instruction or 'N/A'}\n")
                f.write(f"Frames analyzed: {num_frames_to_use}/{len(images)}\n")
                f.write(f"Used state visualization: {use_state_visualization}\n")
                f.write(f"\nVLM Prompt:\n{full_prompt}\n")
                f.write(f"\nVLM Response:\n{vlm_response}\n")
                f.write(f"\nInput image saved as: {traj_name}_input.jpg\n")
        
        return {
            "trajectory_path": trajectory_path,
            "success": True,
            "error": None,
            "vlm_response": vlm_response,
            "vlm_prediction": vlm_prediction,
            "language_instruction": language_instruction,
            "frames_analyzed": num_frames_to_use,
            "total_frames": len(images),
            "used_state_visualization": use_state_visualization
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {trajectory_path}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "trajectory_path": trajectory_path,
            "success": False,
            "error": str(e),
            "vlm_response": None,
            "language_instruction": None
        }


def process_trajectories_parallel(
    trajectory_paths: List[str],
    image_key: str,
    language_key: str,
    question: str,
    max_workers: Optional[int] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple trajectories in parallel with VLM analysis.
    
    Args:
        trajectory_paths: List of paths to trajectory files
        image_key: Key to extract image data (e.g., "observation/images/hand_camera")
        language_key: Key to extract language instruction (e.g., "metadata/language_instruction")
        question: Question to ask the VLM (e.g., "Is this trajectory successful?")
        max_workers: Maximum number of parallel workers (None for automatic)
        
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
    print(f"   Image key: {image_key}")
    print(f"   Language key: {language_key}")
    print(f"   Question: {question}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {output_dir}")
    
    # Submit all tasks to Ray
    futures = []
    for traj_path in trajectory_paths:
        future = process_single_trajectory.remote(
            trajectory_path=traj_path,
            image_key=image_key,
            language_key=language_key,
            question=question,
            tools_config=tools_config,
            output_dir=output_dir
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
            f.write(f"  Image key: {image_key}\n")
            f.write(f"  Language key: {language_key}\n")
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
    # Basic usage
    python simple_vlm_processing.py \\
        --trajectories traj1.h5 traj2.h5 traj3.vla \\
        --image-key "observation/images/hand_camera" \\
        --language-key "metadata/language_instruction" \\
        --question "Is this trajectory successful?"
    
    # Success/failure classification
    python simple_vlm_processing.py \\
        --trajectories /path/to/trajectories/*.h5 \\
        --image-key "observation/images/wrist_camera" \\
        --language-key "metadata/task_description" \\
        --question "Did the robot complete the task successfully?"
    
    # Task understanding
    python simple_vlm_processing.py \\
        --trajectories data/*.vla \\
        --image-key "observation/images/main_camera" \\
        --language-key "instruction" \\
        --question "What task is the robot performing?"
        """)
    
    parser.add_argument(
        "--trajectories", 
        nargs="+", 
        required=True,
        help="Paths to trajectory files (.h5, .hdf5, or .vla)"
    )
    parser.add_argument(
        "--image-key", 
        required=True,
        help="Key to extract image data (e.g., 'observation/images/hand_camera')"
    )
    parser.add_argument(
        "--language-key", 
        required=True,
        help="Key to extract language instruction (e.g., 'metadata/language_instruction')"
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
    
    # Filter for valid trajectory files and check existence
    valid_paths = []
    for path in trajectory_paths:
        if os.path.exists(path):
            ext = os.path.splitext(path.lower())[1]
            if ext in {".h5", ".hdf5", ".vla"}:
                valid_paths.append(path)
            else:
                print(f"⚠️  Skipping {path}: unsupported format (expected .h5, .hdf5, or .vla)")
        else:
            print(f"⚠️  Skipping {path}: file does not exist")
    
    if not valid_paths:
        print("❌ No valid trajectory files found!")
        return 1
    
    print(f"📂 Found {len(valid_paths)} valid trajectory files")
    
    # Process trajectories
    try:
        results = process_trajectories_parallel(
            trajectory_paths=valid_paths,
            image_key=args.image_key,
            language_key=args.language_key,
            question=args.question,
            max_workers=args.max_workers,
            output_dir=args.output_dir
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
                    print(f"   📝 Instruction: {result.get('language_instruction', 'N/A')}")
                    print(f"   🎯 VLM Prediction: {'Success' if result.get('vlm_prediction', False) else 'Failure'}")
                    print(f"   🤖 VLM Response: {result['vlm_response'][:200]}...")
                    print(f"   📊 Frames: {result.get('frames_analyzed', 0)}/{result.get('total_frames', 0)}")
                    if result.get('used_state_visualization', False):
                        print(f"   📈 Used state visualization (no camera images available)")
                else:
                    print(f"   ❌ Error: {result['error']}")
        
        # Print output directory info if used
        if args.output_dir:
            print(f"\n📁 Detailed results saved to: {args.output_dir}/")
            print(f"   - Individual result files: *_results.txt")
            print(f"   - Input images: *_input.jpg")
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