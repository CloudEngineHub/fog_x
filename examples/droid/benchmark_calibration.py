"""
Benchmark for ground truth camera calibration analysis on DROID dataset.

This script analyzes and visualizes ground truth camera calibrations by:
1. Loading calibration data from HuggingFace format (with fallback to other formats)
2. Visualizing end effector trajectories projected using the calibration
3. Verifying that intrinsic and extrinsic matrices are used correctly
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import cv2
import ray
from functools import partial

from robodm.dataset import VLADataset, DatasetConfig


def load_ground_truth_calibration(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ground truth camera calibration data from a trajectory.
    Priority: 1) HuggingFace (hf) extrinsics, 2) Other available extrinsics
    
    Returns:
        Dictionary containing:
        - ground_truth_extrinsics: Ground truth extrinsics for each camera
        - intrinsics: Camera intrinsics if available
        - camera_serials: Camera serial numbers
        - calibration_source: Source of calibration ("hf" or "raw")
        - language_instruction: Task instruction if available
    """
    calibration_data = {
        "ground_truth_extrinsics": {},
        "intrinsics": {},
        "camera_serials": {},
        "serial_to_camera": {},
        "calibration_source": {},
        "language_instruction": ""
    }
    
    # Debug: Print available extrinsic keys
    extrinsic_keys = [k for k in trajectory.keys() if 'camera_extrinsics' in k]
    if extrinsic_keys:
        print(f"Available extrinsic keys (sample): {sorted(extrinsic_keys)[:5]}...")
    
    # Camera names to check
    camera_names = ["wrist", "exterior_image_1", "exterior_image_2"]
    
    # Extract language instruction from various possible locations
    # Try TFDS format first
    if "tfds/steps/language_instruction" in trajectory:
        lang_data = trajectory["tfds/steps/language_instruction"]
        if isinstance(lang_data, (list, np.ndarray)) and len(lang_data) > 0:
            # Take the first instruction
            instruction = lang_data[0]
            if isinstance(instruction, bytes):
                calibration_data["language_instruction"] = instruction.decode("utf-8")
            else:
                calibration_data["language_instruction"] = str(instruction)
    # Try alternative TFDS format
    elif "tfds/observation/language_instruction" in trajectory:
        lang_data = trajectory["tfds/observation/language_instruction"]
        if isinstance(lang_data, (list, np.ndarray)) and len(lang_data) > 0:
            instruction = lang_data[0]
            if isinstance(instruction, bytes):
                calibration_data["language_instruction"] = instruction.decode("utf-8")
            else:
                calibration_data["language_instruction"] = str(instruction)
    # Try raw format
    elif "raw/h5/observation/language_instruction" in trajectory:
        lang_data = trajectory["raw/h5/observation/language_instruction"]
        if isinstance(lang_data, (list, np.ndarray)) and len(lang_data) > 0:
            instruction = lang_data[0]
            if isinstance(instruction, bytes):
                calibration_data["language_instruction"] = instruction.decode("utf-8")
            else:
                calibration_data["language_instruction"] = str(instruction)
    
    # Extract metadata
    metadata_str = trajectory.get("metadata", "")
    if isinstance(metadata_str, (list, np.ndarray)):
        metadata_str = metadata_str[0] if len(metadata_str) > 0 else ""
    
    try:
        metadata = json.loads(metadata_str) if metadata_str else {}
        calibration_data["camera_serials"] = metadata.get("camera_serials", {})
        # Also check metadata for language instruction as fallback
        if not calibration_data["language_instruction"] and "language_instruction" in metadata:
            calibration_data["language_instruction"] = metadata["language_instruction"]
    except:
        metadata = {}
    
    # First, try to get HF calibration as ground truth
    for camera_name in camera_names:
        # Priority 1: HuggingFace extrinsics
        hf_key = f"raw/camera_extrinsics/{camera_name}/hf"
        if hf_key in trajectory:
            hf_data = trajectory[hf_key]
            if isinstance(hf_data, (list, np.ndarray)) and len(hf_data) > 0:
                extrinsic = np.array(hf_data[0]) if hasattr(hf_data[0], '__len__') else np.array(hf_data)
                print(f"  HF extrinsic shape for {camera_name}: {extrinsic.shape}, data: {extrinsic[:10] if len(extrinsic.flatten()) > 10 else extrinsic}")
                # Ensure it's a 4x4 matrix
                if extrinsic.shape == (16,):
                    extrinsic = extrinsic.reshape(4, 4)
                elif extrinsic.shape == (7,):
                    # 7-DOF representation: [x, y, z, qx, qy, qz, qw] (quaternion)
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    quaternion = extrinsic[3:]
                    rotation = Rotation.from_quat(quaternion).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                elif extrinsic.shape == (6,):
                    # 6-DOF representation: [x, y, z, roll, pitch, yaw]
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    rotation = Rotation.from_euler('xyz', extrinsic[3:], degrees=False).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                else:
                    print(f"  WARNING: Unexpected extrinsic shape {extrinsic.shape} for {camera_name}, skipping")
                    continue
                if extrinsic.shape == (4, 4):
                    calibration_data["ground_truth_extrinsics"][camera_name] = extrinsic
                    calibration_data["calibration_source"][camera_name] = "hf"
                    continue
        
        # Priority 2: Raw extrinsics (left) - this is what we actually have!
        raw_key = f"raw/camera_extrinsics/{camera_name}/left"
        if raw_key in trajectory:
            raw_data = trajectory[raw_key]
            if isinstance(raw_data, (list, np.ndarray)) and len(raw_data) > 0:
                # Handle different data structures
                if isinstance(raw_data, np.ndarray):
                    if raw_data.ndim == 1:
                        extrinsic = raw_data
                    elif raw_data.ndim == 2:
                        extrinsic = raw_data[0]
                    else:
                        extrinsic = raw_data.flatten()
                else:
                    extrinsic = np.array(raw_data[0]) if hasattr(raw_data[0], '__len__') else np.array(raw_data)
                print(f"  Raw extrinsic shape for {camera_name}: {extrinsic.shape}, data: {extrinsic[:10] if len(extrinsic.flatten()) > 10 else extrinsic}")
                # Ensure it's a 4x4 matrix
                if extrinsic.shape == (16,):
                    extrinsic = extrinsic.reshape(4, 4)
                elif extrinsic.shape == (7,):
                    # 7-DOF representation: [x, y, z, qx, qy, qz, qw] (quaternion)
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    quaternion = extrinsic[3:]
                    rotation = Rotation.from_quat(quaternion).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                elif extrinsic.shape == (6,):
                    # 6-DOF representation: [x, y, z, roll, pitch, yaw]
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    rotation = Rotation.from_euler('xyz', extrinsic[3:], degrees=False).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                else:
                    print(f"  WARNING: Unexpected extrinsic shape {extrinsic.shape} for {camera_name}, skipping")
                    continue
                if extrinsic.shape == (4, 4):
                    calibration_data["ground_truth_extrinsics"][camera_name] = extrinsic
                    calibration_data["calibration_source"][camera_name] = "raw"
                    print(f"  Found calibration for {camera_name} at {raw_key}, final shape: {extrinsic.shape}")
                    continue
                else:
                    print(f"  ERROR: Extrinsic conversion failed for {camera_name}, shape is {extrinsic.shape} instead of (4, 4)")
                    
        # Priority 3: Check H5 keys with camera name (e.g., raw/h5/observation/camera_extrinsics/wrist_left)
        h5_key = f"raw/h5/observation/camera_extrinsics/{camera_name}_left"
        if h5_key in trajectory:
            h5_data = trajectory[h5_key]
            if isinstance(h5_data, (list, np.ndarray)) and len(h5_data) > 0:
                extrinsic = np.array(h5_data[0]) if hasattr(h5_data[0], '__len__') else np.array(h5_data)
                print(f"  H5 extrinsic shape for {camera_name}: {extrinsic.shape}, data: {extrinsic[:10] if len(extrinsic.flatten()) > 10 else extrinsic}")
                # Ensure it's a 4x4 matrix
                if extrinsic.shape == (16,):
                    extrinsic = extrinsic.reshape(4, 4)
                elif extrinsic.shape == (7,):
                    # 7-DOF representation: [x, y, z, qx, qy, qz, qw] (quaternion)
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    quaternion = extrinsic[3:]
                    rotation = Rotation.from_quat(quaternion).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                elif extrinsic.shape == (6,):
                    # 6-DOF representation: [x, y, z, roll, pitch, yaw]
                    # Convert to 4x4 matrix
                    from scipy.spatial.transform import Rotation
                    translation = extrinsic[:3]
                    rotation = Rotation.from_euler('xyz', extrinsic[3:], degrees=False).as_matrix()
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rotation
                    extrinsic[:3, 3] = translation
                else:
                    print(f"  WARNING: Unexpected extrinsic shape {extrinsic.shape} for {camera_name}, skipping")
                    continue
                if extrinsic.shape == (4, 4):
                    calibration_data["ground_truth_extrinsics"][camera_name] = extrinsic
                    calibration_data["calibration_source"][camera_name] = "h5"
                    continue
    
    # Also check for serial-based keys that weren't renamed
    all_extrinsic_keys = [k for k in trajectory.keys() if 'camera_extrinsics' in k and '_left' in k]
    
    # Create a mapping of all serials found in H5 to potential camera names
    # This handles cases where metadata might be missing or incomplete
    unmapped_serials = []
    
    for key in all_extrinsic_keys:
        parts = key.split('/')
        if len(parts) > 0:
            serial_part = parts[-1]  # e.g., '18026681_left' or 'wrist_left'
            
            # Check if this is already a camera name
            is_camera_name = False
            for camera_name in camera_names:
                if serial_part.startswith(camera_name + "_"):
                    is_camera_name = True
                    break
            
            if not is_camera_name:
                # This is likely a serial number
                serial = serial_part.replace('_left', '').replace('_right', '')
                if serial.isdigit():
                    # Try to match serial to camera name from metadata
                    matched = False
                    for camera_name in camera_names:
                        if camera_name in calibration_data["camera_serials"] and str(calibration_data["camera_serials"][camera_name]) == serial:
                            calibration_data["serial_to_camera"][serial] = camera_name
                            # Get the extrinsic data if we don't have it yet
                            if camera_name not in calibration_data["ground_truth_extrinsics"]:
                                extrinsic_data = trajectory.get(key)
                                if isinstance(extrinsic_data, (list, np.ndarray)) and len(extrinsic_data) > 0:
                                    extrinsic = np.array(extrinsic_data[0]) if hasattr(extrinsic_data[0], '__len__') else np.array(extrinsic_data)
                                    # Ensure it's a 4x4 matrix
                                    if extrinsic.shape == (16,):
                                        extrinsic = extrinsic.reshape(4, 4)
                                    elif extrinsic.shape == (7,):
                                        # 7-DOF representation: [x, y, z, qx, qy, qz, qw] (quaternion)
                                        # Convert to 4x4 matrix
                                        from scipy.spatial.transform import Rotation
                                        translation = extrinsic[:3]
                                        quaternion = extrinsic[3:]
                                        rotation = Rotation.from_quat(quaternion).as_matrix()
                                        extrinsic = np.eye(4)
                                        extrinsic[:3, :3] = rotation
                                        extrinsic[:3, 3] = translation
                                    elif extrinsic.shape == (6,):
                                        # Convert from 6-DOF representation to 4x4 matrix
                                        continue  # Skip this format for now
                                    if extrinsic.shape == (4, 4):
                                        calibration_data["ground_truth_extrinsics"][camera_name] = extrinsic
                                        calibration_data["calibration_source"][camera_name] = "serial"
                            matched = True
                            break
                    
                    if not matched:
                        unmapped_serials.append(serial)
    
    # If we have unmapped serials and missing cameras, try to make educated guesses
    if unmapped_serials and len(calibration_data["ground_truth_extrinsics"]) < len(camera_names):
        print(f"⚠️  Found unmapped serials: {unmapped_serials}")
        missing_cameras = [cam for cam in camera_names if cam not in calibration_data["ground_truth_extrinsics"]]
        print(f"⚠️  Missing cameras: {missing_cameras}")
        print(f"⚠️  Found calibration for: {list(calibration_data['ground_truth_extrinsics'].keys())}")
    
    # Get intrinsics if available
    intrinsic_keys = [k for k in trajectory.keys() if 'camera_intrinsics' in k]
    if intrinsic_keys:
        print(f"Available intrinsic keys: {sorted(intrinsic_keys)[:10]}...")
    
    for camera_name in camera_names:
        intrinsic_key = f"raw/camera_intrinsics/{camera_name}"
        if intrinsic_key in trajectory:
            intrinsic_data = trajectory[intrinsic_key]
            if isinstance(intrinsic_data, (list, np.ndarray)) and len(intrinsic_data) > 0:
                # Handle both single matrix and array of matrices
                if isinstance(intrinsic_data, np.ndarray):
                    if intrinsic_data.ndim == 3 and intrinsic_data.shape[0] > 0:
                        # Array of matrices, take first one
                        intrinsic_matrix = intrinsic_data[0]
                    elif intrinsic_data.ndim == 2 and intrinsic_data.shape == (3, 3):
                        # Single matrix
                        intrinsic_matrix = intrinsic_data
                    else:
                        # Flatten and reshape if needed
                        intrinsic_matrix = np.array(intrinsic_data).reshape(3, 3)
                else:
                    intrinsic_matrix = np.array(intrinsic_data[0]) if hasattr(intrinsic_data[0], '__len__') else np.array(intrinsic_data)
                
                # Ensure it's 3x3
                if intrinsic_matrix.shape != (3, 3):
                    intrinsic_matrix = intrinsic_matrix.reshape(3, 3)
                
                calibration_data["intrinsics"][camera_name] = intrinsic_matrix
                print(f"  Loaded intrinsics for {camera_name}, shape: {intrinsic_matrix.shape}")
    
    return calibration_data




def project_point_to_image(point_3d: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> Tuple[int, int]:
    """
    Project a 3D point to 2D image coordinates using camera calibration.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        extrinsic: 4x4 camera extrinsic matrix (transforms from world to camera coordinates)
        intrinsic: 3x3 camera intrinsic matrix
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    # Validate inputs
    if len(point_3d) != 3:
        print(f"ERROR: point_3d has {len(point_3d)} elements, expected 3")
        return -1, -1
    if extrinsic.shape != (4, 4):
        print(f"ERROR: extrinsic has shape {extrinsic.shape}, expected (4, 4)")
        return -1, -1
    if intrinsic.shape != (3, 3):
        print(f"ERROR: intrinsic has shape {intrinsic.shape}, expected (3, 3)")
        return -1, -1
    
    # Convert to homogeneous coordinates
    point_3d_homo = np.append(point_3d, 1)
    
    # Transform from world to camera coordinates using the inverse of extrinsic
    # The extrinsic matrix typically represents camera pose in world coordinates
    # To transform points from world to camera, we need the inverse
    try:
        extrinsic_inv = np.linalg.inv(extrinsic)
        point_cam = extrinsic_inv @ point_3d_homo
    except:
        # If inverse fails, assume extrinsic is already world-to-camera transform
        point_cam = extrinsic @ point_3d_homo
    
    # Project to image plane
    if point_cam[2] > 0:  # Point is in front of camera
        point_2d = intrinsic @ point_cam[:3]
        point_2d = point_2d / point_2d[2]
        return int(point_2d[0]), int(point_2d[1])
    else:
        return -1, -1  # Point behind camera


def visualize_ground_truth_calibration(
    trajectory: Dict[str, Any],
    ground_truth_extrinsic: np.ndarray,
    camera_name: str,
    intrinsic: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    language_instruction: str = ""
) -> np.ndarray:
    """
    Visualize end effector trajectory using ground truth calibration.
    
    Returns:
        Visualization image showing the ground truth calibration
    """
    if intrinsic is None:
        print(f"Warning: No intrinsic matrix found for {camera_name}, using default")
        # Create a default intrinsic matrix based on ZED camera typical parameters
        # This matches the default intrinsics from the DROID dataset
        intrinsic = np.array([
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ])
    else:
        print(f"  Using stored intrinsics for {camera_name}, shape: {intrinsic.shape}")
        
    # Get camera images
    image_key = f"raw/images/{camera_name}_left"
    if image_key not in trajectory:
        # Try TFDS format
        image_key = f"tfds/observation/images/{camera_name}"
    
    if image_key not in trajectory:
        # Try to find any image key that might match
        for k in trajectory.keys():
            if 'images' in k and camera_name in k:
                image_key = k
                break
    
    if image_key not in trajectory:
        return None
    
    images = trajectory[image_key]
    if len(images) == 0:
        return None
    
    # Get end effector positions
    ee_pos_key = "raw/h5/observation/robot_state/cartesian_position"
    if ee_pos_key not in trajectory:
        ee_pos_key = "tfds/observation/cartesian_position"  # Try TFDS format
    if ee_pos_key not in trajectory:
        ee_pos_key = "tfds/observation/state"  # Try another TFDS format
    
    if ee_pos_key not in trajectory:
        print(f"Warning: No end effector position data found for {camera_name}")
        return None
    
    ee_positions = trajectory[ee_pos_key]
    
    # Check if we have valid position data
    if len(ee_positions) == 0:
        print(f"Warning: Empty end effector position data for {camera_name}")
        return None
    
    # Select multiple frames throughout the trajectory to show the trajectory
    num_frames = min(10, len(images))  # Show up to 10 points along trajectory
    frame_indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
    
    # Use the middle frame as the base image
    base_frame_idx = len(images) // 2
    visualization_frame = images[base_frame_idx].copy()
    
    # Validate extrinsic matrix
    if ground_truth_extrinsic.shape != (4, 4):
        print(f"ERROR: ground_truth_extrinsic has shape {ground_truth_extrinsic.shape}, expected (4, 4)")
        return None
    
    # Draw the end effector trajectory across multiple frames
    trajectory_points = []
    for frame_idx in frame_indices:
        if frame_idx < len(ee_positions):
            ee_pos_raw = ee_positions[frame_idx]
            
            # Handle different position formats
            if isinstance(ee_pos_raw, (list, np.ndarray)):
                if len(ee_pos_raw) >= 7:
                    # 7-element format: [x, y, z, qx, qy, qz, qw]
                    ee_pos = ee_pos_raw[:3]
                elif len(ee_pos_raw) == 6:
                    # 6-element format: [x, y, z, roll, pitch, yaw]
                    ee_pos = ee_pos_raw[:3]
                elif len(ee_pos_raw) == 3:
                    # Already just position
                    ee_pos = ee_pos_raw
                else:
                    print(f"Warning: Unexpected ee_pos shape: {len(ee_pos_raw)}")
                    continue
            else:
                print(f"Warning: Unexpected ee_pos type: {type(ee_pos_raw)}")
                continue
            
            # Ensure ee_pos is a numpy array with 3 elements
            ee_pos = np.array(ee_pos)[:3]
            
            # Project using ground truth calibration
            px, py = project_point_to_image(ee_pos, ground_truth_extrinsic, intrinsic)
            if px >= 0 and py >= 0 and px < visualization_frame.shape[1] and py < visualization_frame.shape[0]:
                trajectory_points.append((px, py))
                # Draw circle for each point
                cv2.circle(visualization_frame, (px, py), 5, (0, 255, 0), -1)  # Green filled circle
    
    # Draw lines connecting the trajectory points
    if len(trajectory_points) > 1:
        for i in range(len(trajectory_points) - 1):
            cv2.line(visualization_frame, trajectory_points[i], trajectory_points[i+1], (0, 255, 0), 2)
    
    # Add labels
    cv2.putText(visualization_frame, f"Ground Truth Calibration - {camera_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(visualization_frame, f"End Effector Trajectory (Green)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add language instruction if available
    if language_instruction:
        # Wrap long text
        max_width = 60  # characters per line
        words = language_instruction.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_width:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw task instruction
        y_offset = 90
        cv2.putText(visualization_frame, "Task:", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, line in enumerate(lines[:3]):  # Limit to 3 lines
            cv2.putText(visualization_frame, line, (10, y_offset + 25 * (i + 1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(visualization_frame, cv2.COLOR_RGB2BGR))
    
    return visualization_frame




def process_single_trajectory(
    trajectory: Dict[str, Any], 
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process a single trajectory and visualize ground truth calibration.
    """
    file_path = trajectory.get("__file_path__", "")
    traj_name = Path(file_path).stem
    
    print(f"\n📐 Processing {traj_name}")
    
    # Load ground truth calibration
    calibration_data = load_ground_truth_calibration(trajectory)
    
    # Display language instruction if available
    if calibration_data.get("language_instruction"):
        print(f"  Task: {calibration_data['language_instruction']}")
    else:
        print(f"  Task: No language instruction found")
    
    # Results for this trajectory
    results = {
        "trajectory_name": traj_name,
        "language_instruction": calibration_data.get("language_instruction", ""),
        "camera_evaluations": {},
        "has_calibration": len(calibration_data["ground_truth_extrinsics"]) > 0
    }
    
    if not results["has_calibration"]:
        print(f"⚠️  No calibration data found for {traj_name}")
        return results
    
    # Process all cameras
    for camera_name in calibration_data["ground_truth_extrinsics"].keys():
        camera_results = {
            "has_calibration": True,
            "calibration_source": calibration_data["calibration_source"].get(camera_name, "unknown"),
            "camera_serial": calibration_data["camera_serials"].get(camera_name, "unknown")
        }
        
        # Get ground truth calibration
        gt_extrinsic = calibration_data["ground_truth_extrinsics"][camera_name]
        
        # Ensure gt_extrinsic is a proper 4x4 matrix
        if gt_extrinsic.shape != (4, 4):
            print(f"  ERROR: Ground truth extrinsic for {camera_name} has shape {gt_extrinsic.shape}, expected (4, 4)")
            continue
            
        intrinsic = calibration_data["intrinsics"].get(camera_name)
        
        # Print calibration info
        print(f"\n  Camera: {camera_name}")
        print(f"  Calibration source: {camera_results['calibration_source']}")
        print(f"  Camera serial: {camera_results['camera_serial']}")
        print(f"  Has intrinsics: {'Yes' if intrinsic is not None else 'No'}")
        
        # Print extrinsic matrix
        print(f"  Extrinsic matrix:")
        print(f"    Rotation:")
        for i in range(3):
            print(f"      [{gt_extrinsic[i, 0]:7.4f}, {gt_extrinsic[i, 1]:7.4f}, {gt_extrinsic[i, 2]:7.4f}]")
        print(f"    Translation: [{gt_extrinsic[0, 3]:7.4f}, {gt_extrinsic[1, 3]:7.4f}, {gt_extrinsic[2, 3]:7.4f}]")
        
        if intrinsic is not None:
            print(f"  Intrinsic matrix:")
            print(f"    fx: {intrinsic[0, 0]:.2f}, fy: {intrinsic[1, 1]:.2f}")
            print(f"    cx: {intrinsic[0, 2]:.2f}, cy: {intrinsic[1, 2]:.2f}")
        
        # Generate visualization
        vis_path = output_dir / f"{traj_name}_{camera_name}_calibration.jpg"
        vis_image = visualize_ground_truth_calibration(
            trajectory, gt_extrinsic, camera_name, intrinsic, vis_path,
            language_instruction=calibration_data.get("language_instruction", "")
        )
        
        if vis_image is not None:
            camera_results["visualization_saved"] = True
            print(f"  Visualization saved to: {vis_path}")
        else:
            camera_results["visualization_saved"] = False
            print(f"  WARNING: Could not generate visualization")
        
        results["camera_evaluations"][camera_name] = camera_results
    
    # Save detailed results
    results_file = output_dir / f"{traj_name}_calibration_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


class CalibrationVisualizationBenchmark:
    """Benchmark for visualizing and analyzing ground truth camera calibrations."""
    
    def __init__(self, dataset_path: str, output_dir: str = "./calibration_benchmark_results"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = DatasetConfig(
            batch_size=4,
            shuffle=False,
            use_metadata=False,
            auto_build_metadata=False
        )
    
    def load_dataset(self, max_trajectories: Optional[int] = None) -> VLADataset:
        """Load the VLA dataset."""
        print(f"Loading dataset from: {self.dataset_path}")
        
        dataset = VLADataset(
            path=self.dataset_path,
            return_type="numpy",
            config=self.config
        )
        
        total_trajectories = dataset.count()
        print(f"Found {total_trajectories} trajectory files")
        
        if max_trajectories is not None and total_trajectories > max_trajectories:
            print(f"Limiting to {max_trajectories} trajectories")
            limited_items = dataset.take(max_trajectories)
            
            if limited_items:
                limited_file_paths = [item if isinstance(item, str) else item.get("item", str(item)) 
                                    for item in limited_items]
                
                import ray.data as rd
                limited_ray_dataset = rd.from_items(limited_file_paths)
                
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
    
    def run_benchmark(self, max_trajectories: Optional[int] = None) -> Dict[str, Any]:
        """Run the calibration visualization benchmark."""
        print("\n" + "=" * 60)
        print("GROUND TRUTH CAMERA CALIBRATION ANALYSIS")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset(max_trajectories)
        
        # Process trajectories
        process_fn = partial(
            process_single_trajectory, 
            output_dir=self.output_dir
        )
        results_dataset = dataset.map(process_fn).materialize()
        results = list(results_dataset.iter_rows())
        
        # Aggregate results
        total_trajectories = len(results)
        trajectories_with_calibration = 0
        total_cameras = 0
        cameras_by_source = {"hf": 0, "raw": 0, "h5": 0, "serial": 0, "unknown": 0}
        cameras_with_intrinsics = 0
        cameras_with_visualization = 0
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in results:
            if result["has_calibration"]:
                trajectories_with_calibration += 1
                
                num_cameras = len(result["camera_evaluations"])
                for camera_name, camera_eval in result["camera_evaluations"].items():
                    total_cameras += 1
                    
                    # Count calibration sources
                    source = camera_eval.get("calibration_source", "unknown")
                    if source in cameras_by_source:
                        cameras_by_source[source] += 1
                    else:
                        cameras_by_source["unknown"] += 1
                    
                    # Count visualizations
                    if camera_eval.get("visualization_saved", False):
                        cameras_with_visualization += 1
                
                print(f"✅ {result['trajectory_name']}: {num_cameras} cameras with calibration")
        
        print(f"\nBenchmark Summary:")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Trajectories with calibration: {trajectories_with_calibration}")
        print(f"Total cameras evaluated: {total_cameras}")
        print(f"\nCalibration sources:")
        for source, count in cameras_by_source.items():
            if count > 0:
                print(f"  {source}: {count} ({count/total_cameras*100:.1f}%)")
        print(f"\nCameras with visualization: {cameras_with_visualization} ({cameras_with_visualization/total_cameras*100:.1f}%)")
        
        # Save summary
        summary = {
            "total_trajectories": total_trajectories,
            "trajectories_with_calibration": trajectories_with_calibration,
            "total_cameras": total_cameras,
            "cameras_by_source": cameras_by_source,
            "cameras_with_visualization": cameras_with_visualization
        }
        
        summary_file = self.output_dir / "calibration_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {self.output_dir}/")
        
        return summary


def main():
    """Main function to run the ground truth calibration analysis."""
    parser = argparse.ArgumentParser(description="Analyze and visualize ground truth camera calibrations in DROID dataset")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./droid_combined_data",
        help="Path to the directory containing VLA trajectory files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./calibration_benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=100,
        help="Maximum number of trajectories to process"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create and run benchmark
        benchmark = CalibrationVisualizationBenchmark(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        summary = benchmark.run_benchmark(max_trajectories=args.max_trajectories)
        
        print(f"\nAnalysis complete!")
        print(f"Total cameras analyzed: {summary['total_cameras']}")
        print(f"Visualizations generated: {summary['cameras_with_visualization']}")
        
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()