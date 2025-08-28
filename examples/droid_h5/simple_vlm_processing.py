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
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from robodm import Trajectory
from robodm.agent.tools import ToolsManager


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
    tools_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single trajectory with VLM analysis.
    
    Args:
        trajectory_path: Path to the trajectory file (.h5, .hdf5, or .vla)
        image_key: Key to extract image data from trajectory
        language_key: Key to extract language instruction from trajectory  
        question: Question to ask the VLM
        tools_config: Configuration for VLM tools
        
    Returns:
        Dictionary with trajectory analysis results
    """
    try:
        print(f"🔄 Processing {os.path.basename(trajectory_path)}")
        
        # Load trajectory
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
        images = None
        use_state_visualization = False
        
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
        language_instruction = None
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
        
        # Prepare VLM prompt
        context = f"\nLanguage instruction: '{language_instruction}'" if language_instruction else ""
        
        if use_state_visualization:
            full_prompt = f"""Analyze these {num_frames_to_use} robot state visualizations and answer: {question}

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

You must choose either "Yes" (successful) or "No" (failed). Do not hedge. Be critical - if you see any signs the robot didn't complete its intended task, answer "No".

Answer with a clear Yes or No first, then explain your reasoning based on task completion evidence.{context}"""
        else:
            full_prompt = f"{question}{context}\n\nPlease analyze these {num_frames_to_use} frames from the robot trajectory and provide a clear answer."
        
        # Call VLM
        vlm_response = vlm_tool(grid_image, full_prompt)
        
        print(f"  ✅ VLM Response: '{vlm_response[:100]}...'")
        
        return {
            "trajectory_path": trajectory_path,
            "success": True,
            "error": None,
            "vlm_response": vlm_response,
            "language_instruction": language_instruction,
            "frames_analyzed": num_frames_to_use,
            "total_frames": len(images)
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
    max_workers: Optional[int] = None
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
    
    # Submit all tasks to Ray
    futures = []
    for traj_path in trajectory_paths:
        future = process_single_trajectory.remote(
            trajectory_path=traj_path,
            image_key=image_key,
            language_key=language_key,
            question=question,
            tools_config=tools_config
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
    successful = sum(1 for r in results.values() if r["success"])
    failed = len(results) - successful
    
    print(f"\n📈 Processing Complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Rate: {len(trajectory_paths)/total_time*60:.1f} trajectories/minute")
    
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
            max_workers=args.max_workers
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
                    print(f"   🤖 VLM Response: {result['vlm_response']}")
                    print(f"   📊 Frames: {result.get('frames_analyzed', 0)}/{result.get('total_frames', 0)}")
                else:
                    print(f"   ❌ Error: {result['error']}")
        
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