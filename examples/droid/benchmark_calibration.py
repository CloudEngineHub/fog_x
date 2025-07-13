"""
Benchmark for camera calibration evaluation using VLM on DROID dataset.

This script evaluates VLM's ability to correct camera calibration errors by:
1. Using HuggingFace calibration as ground truth (with fallback to other extrinsics)
2. Introducing synthetic calibration errors at a fixed rate
3. Asking VLM to identify and suggest corrections for calibration errors
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
from robodm.agent.vlm_service import get_vlm_service


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
    """
    calibration_data = {
        "ground_truth_extrinsics": {},
        "intrinsics": {},
        "camera_serials": {},
        "serial_to_camera": {},
        "calibration_source": {}
    }
    
    # Debug: Print available extrinsic keys
    extrinsic_keys = [k for k in trajectory.keys() if 'camera_extrinsics' in k]
    if extrinsic_keys:
        print(f"Available extrinsic keys (sample): {sorted(extrinsic_keys)[:5]}...")
    
    # Camera names to check
    camera_names = ["wrist", "exterior_image_1", "exterior_image_2"]
    
    # Extract metadata
    metadata_str = trajectory.get("metadata", "")
    if isinstance(metadata_str, (list, np.ndarray)):
        metadata_str = metadata_str[0] if len(metadata_str) > 0 else ""
    
    try:
        metadata = json.loads(metadata_str) if metadata_str else {}
        calibration_data["camera_serials"] = metadata.get("camera_serials", {})
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
    for camera_name in camera_names:
        intrinsic_key = f"raw/camera_intrinsics/{camera_name}"
        if intrinsic_key in trajectory:
            intrinsic_data = trajectory[intrinsic_key]
            if isinstance(intrinsic_data, (list, np.ndarray)) and len(intrinsic_data) > 0:
                calibration_data["intrinsics"][camera_name] = np.array(intrinsic_data[0]) if hasattr(intrinsic_data[0], '__len__') else np.array(intrinsic_data)
    
    return calibration_data


def corrupt_calibration(extrinsic: np.ndarray, corruption_type: str = "rotation_translation") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Introduce synthetic calibration errors to an extrinsic matrix.
    
    Args:
        extrinsic: 4x4 camera extrinsic matrix
        corruption_type: Type of corruption ("rotation", "translation", "rotation_translation")
        
    Returns:
        Tuple of (corrupted_extrinsic, corruption_params)
    """
    if extrinsic.shape != (4, 4):
        return extrinsic, {"error": "Invalid extrinsic shape"}
    
    corrupted = extrinsic.copy()
    corruption_params = {"type": corruption_type}
    
    if corruption_type in ["rotation", "rotation_translation"]:
        # Add rotation error (5-15 degrees around random axis)
        angle = np.random.uniform(5, 15)  # degrees
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        angle_rad = np.radians(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_error = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K
        
        # Apply rotation error
        corrupted[:3, :3] = R_error @ extrinsic[:3, :3]
        corruption_params["rotation_angle"] = angle
        corruption_params["rotation_axis"] = axis.tolist()
    
    if corruption_type in ["translation", "rotation_translation"]:
        # Add translation error (0.05-0.15 meters in random direction)
        magnitude = np.random.uniform(0.05, 0.15)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        translation_error = magnitude * direction
        
        # Apply translation error
        corrupted[:3, 3] = extrinsic[:3, 3] + translation_error
        corruption_params["translation_magnitude"] = magnitude
        corruption_params["translation_direction"] = direction.tolist()
    
    return corrupted, corruption_params


def project_point_to_image(point_3d: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> Tuple[int, int]:
    """
    Project a 3D point to 2D image coordinates using camera calibration.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        extrinsic: 4x4 camera extrinsic matrix
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
    
    # Convert to homogeneous coordinates
    point_3d_homo = np.append(point_3d, 1)
    
    # Transform to camera coordinates
    point_cam = extrinsic @ point_3d_homo
    
    # Project to image plane
    if point_cam[2] > 0:  # Point is in front of camera
        point_2d = intrinsic @ point_cam[:3]
        point_2d = point_2d / point_2d[2]
        return int(point_2d[0]), int(point_2d[1])
    else:
        return -1, -1  # Point behind camera


def visualize_calibration_comparison(
    trajectory: Dict[str, Any],
    ground_truth_extrinsic: np.ndarray,
    corrupted_extrinsic: np.ndarray,
    camera_name: str,
    intrinsic: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Visualize end effector trajectory using both ground truth and corrupted calibration.
    
    Returns:
        Visualization image showing both calibrations side by side
    """
    if intrinsic is None:
        print(f"Warning: No intrinsic matrix found for {camera_name}, using default")
        # Create a default intrinsic matrix based on typical image size
        # Assuming 640x480 image with focal length ~500
        intrinsic = np.array([
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ])
        
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
    
    # Select a frame from the middle of the trajectory
    frame_idx = len(images) // 2
    base_frame = images[frame_idx].copy()
    
    # Create two copies for visualization
    gt_frame = base_frame.copy()
    corrupted_frame = base_frame.copy()
    
    # Validate extrinsic matrices
    if ground_truth_extrinsic.shape != (4, 4):
        print(f"ERROR: ground_truth_extrinsic has shape {ground_truth_extrinsic.shape}, expected (4, 4)")
        return None
    if corrupted_extrinsic.shape != (4, 4):
        print(f"ERROR: corrupted_extrinsic has shape {corrupted_extrinsic.shape}, expected (4, 4)")
        return None
    
    # Draw only the current frame's end effector position
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
                return None
        else:
            print(f"Warning: Unexpected ee_pos type: {type(ee_pos_raw)}")
            return None
        
        # Ensure ee_pos is a numpy array with 3 elements
        ee_pos = np.array(ee_pos)[:3]
        
        # Project using ground truth calibration
        gt_px, gt_py = project_point_to_image(ee_pos, ground_truth_extrinsic, intrinsic)
        if gt_px >= 0 and gt_py >= 0 and gt_px < gt_frame.shape[1] and gt_py < gt_frame.shape[0]:
            # Draw a larger circle for better visibility
            cv2.circle(gt_frame, (gt_px, gt_py), 8, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(gt_frame, (gt_px, gt_py), 10, (0, 255, 0), 2)  # Green outline
        
        # Project using corrupted calibration
        corr_px, corr_py = project_point_to_image(ee_pos, corrupted_extrinsic, intrinsic)
        if corr_px >= 0 and corr_py >= 0 and corr_px < corrupted_frame.shape[1] and corr_py < corrupted_frame.shape[0]:
            # Draw a larger circle for better visibility
            cv2.circle(corrupted_frame, (corr_px, corr_py), 8, (255, 0, 0), -1)  # Red filled circle
            cv2.circle(corrupted_frame, (corr_px, corr_py), 10, (255, 0, 0), 2)  # Red outline
    
    # Add labels
    cv2.putText(gt_frame, "Ground Truth (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(corrupted_frame, "Corrupted (Red)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Combine frames side by side
    combined = np.hstack([gt_frame, corrupted_frame])
    
    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return combined


def compute_transformation_difference(T1: np.ndarray, T2: np.ndarray) -> Dict[str, Any]:
    """
    Compute the transformation difference between two 4x4 matrices.
    Returns the transformation that would correct T2 to match T1.
    """
    if T1.shape != (4, 4) or T2.shape != (4, 4):
        return {"error": "Invalid transformation shape"}
    
    # Compute T_correction such that T1 = T_correction @ T2
    T_correction = T1 @ np.linalg.inv(T2)
    
    # Extract rotation and translation
    R_correction = T_correction[:3, :3]
    t_correction = T_correction[:3, 3]
    
    # Convert rotation to axis-angle
    trace = np.trace(R_correction)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if np.abs(angle) < 1e-6:
        axis = np.array([0, 0, 1])  # Arbitrary axis for zero rotation
    else:
        axis = np.array([
            R_correction[2, 1] - R_correction[1, 2],
            R_correction[0, 2] - R_correction[2, 0],
            R_correction[1, 0] - R_correction[0, 1]
        ])
        axis = axis / (2 * np.sin(angle))
    
    return {
        "rotation_angle_deg": np.degrees(angle),
        "rotation_axis": axis.tolist(),
        "translation": t_correction.tolist(),
        "correction_matrix": T_correction.tolist()
    }


def process_single_trajectory_with_corruption(
    trajectory: Dict[str, Any], 
    output_dir: Path,
    corruption_rate: float = 0.5
) -> Dict[str, Any]:
    """
    Process a single trajectory with synthetic calibration corruption.
    """
    file_path = trajectory.get("__file_path__", "")
    traj_name = Path(file_path).stem
    
    print(f"\n📐 Processing {traj_name} with corruption rate {corruption_rate}")
    
    # Load ground truth calibration
    calibration_data = load_ground_truth_calibration(trajectory)
    
    # Results for this trajectory
    results = {
        "trajectory_name": traj_name,
        "camera_evaluations": {},
        "has_calibration": len(calibration_data["ground_truth_extrinsics"]) > 0
    }
    
    if not results["has_calibration"]:
        print(f"⚠️  No calibration data found for {traj_name}")
        return results
    
    # Process only side cameras (no wrist)
    for camera_name in ["exterior_image_1", "exterior_image_2"]:
        if camera_name not in calibration_data["ground_truth_extrinsics"]:
            continue
        
        camera_results = {
            "has_calibration": True,
            "calibration_source": calibration_data["calibration_source"].get(camera_name, "unknown"),
            "was_corrupted": False,
            "corruption_params": None,
            "vlm_evaluation": None,
            "ground_truth_correction": None
        }
        
        # Get ground truth calibration
        gt_extrinsic = calibration_data["ground_truth_extrinsics"][camera_name]
        
        # Ensure gt_extrinsic is a proper 4x4 matrix
        if gt_extrinsic.shape != (4, 4):
            print(f"  ERROR: Ground truth extrinsic for {camera_name} has shape {gt_extrinsic.shape}, expected (4, 4)")
            continue
            
        intrinsic = calibration_data["intrinsics"].get(camera_name)
        
        # Decide whether to corrupt this camera's calibration
        if np.random.random() < corruption_rate:
            camera_results["was_corrupted"] = True
            
            # Create corrupted calibration
            corrupted_extrinsic, corruption_params = corrupt_calibration(gt_extrinsic)
            camera_results["corruption_params"] = corruption_params
            
            # Compute ground truth correction
            camera_results["ground_truth_correction"] = compute_transformation_difference(
                gt_extrinsic, corrupted_extrinsic
            )
            
            # Generate visualization
            vis_path = output_dir / f"{traj_name}_{camera_name}_calibration_comparison.jpg"
            vis_image = visualize_calibration_comparison(
                trajectory, gt_extrinsic, corrupted_extrinsic, 
                camera_name, intrinsic, vis_path
            )
            
            # Use VLM to evaluate and suggest correction
            if vis_image is not None:
                try:
                    vlm_service = get_vlm_service()
                    vlm_service.initialize()
                    
                    vlm_prompt = """You are analyzing robot camera calibration. The image shows:
- Left: Robot end effector trajectory with CORRECT calibration (green dots/lines)
- Right: Same trajectory with INCORRECT calibration (red dots/lines)

The incorrect calibration has rotation and/or translation errors.

Please analyze the calibration error and provide the transformation needed to correct it:

1. ROTATION_ERROR: Estimate the rotation error in degrees and the axis of rotation
2. TRANSLATION_ERROR: Estimate the translation error in meters and direction
3. CONFIDENCE: Your confidence in the estimates (HIGH/MEDIUM/LOW)

Format your response as:
ROTATION_ANGLE: [degrees]
ROTATION_AXIS: [x, y, z] (normalized)
TRANSLATION_MAGNITUDE: [meters]
TRANSLATION_DIRECTION: [x, y, z] (normalized)
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Brief explanation of what you observe]"""
                    
                    vlm_response = vlm_service.analyze_image(vis_image, vlm_prompt)
                    
                    # Parse VLM response
                    vlm_eval = {
                        "raw_response": vlm_response,
                        "rotation_angle": None,
                        "rotation_axis": None,
                        "translation_magnitude": None,
                        "translation_direction": None,
                        "confidence": "UNKNOWN",
                        "explanation": ""
                    }
                    
                    lines = vlm_response.strip().split('\n')
                    for line in lines:
                        if "ROTATION_ANGLE:" in line:
                            try:
                                vlm_eval["rotation_angle"] = float(line.split(":")[-1].strip())
                            except:
                                pass
                        elif "ROTATION_AXIS:" in line:
                            try:
                                axis_str = line.split(":")[-1].strip()
                                axis = eval(axis_str)  # Parse list
                                vlm_eval["rotation_axis"] = axis
                            except:
                                pass
                        elif "TRANSLATION_MAGNITUDE:" in line:
                            try:
                                vlm_eval["translation_magnitude"] = float(line.split(":")[-1].strip())
                            except:
                                pass
                        elif "TRANSLATION_DIRECTION:" in line:
                            try:
                                dir_str = line.split(":")[-1].strip()
                                direction = eval(dir_str)  # Parse list
                                vlm_eval["translation_direction"] = direction
                            except:
                                pass
                        elif "CONFIDENCE:" in line:
                            vlm_eval["confidence"] = line.split(":")[-1].strip()
                        elif "EXPLANATION:" in line:
                            vlm_eval["explanation"] = line.split(":", 1)[-1].strip()
                    
                    camera_results["vlm_evaluation"] = vlm_eval
                    
                    # Calculate VLM accuracy
                    if vlm_eval["rotation_angle"] is not None and camera_results["ground_truth_correction"]:
                        gt_angle = camera_results["ground_truth_correction"]["rotation_angle_deg"]
                        vlm_angle = vlm_eval["rotation_angle"]
                        angle_error = abs(gt_angle - vlm_angle)
                        camera_results["vlm_rotation_error"] = angle_error
                    
                    if vlm_eval["translation_magnitude"] is not None and camera_results["corruption_params"]:
                        gt_magnitude = camera_results["corruption_params"].get("translation_magnitude", 0)
                        vlm_magnitude = vlm_eval["translation_magnitude"]
                        magnitude_error = abs(gt_magnitude - vlm_magnitude)
                        camera_results["vlm_translation_error"] = magnitude_error
                    
                except Exception as e:
                    print(f"VLM evaluation failed for {camera_name}: {e}")
                    camera_results["vlm_evaluation"] = {"error": str(e)}
        
        results["camera_evaluations"][camera_name] = camera_results
    
    # Save detailed results
    results_file = output_dir / f"{traj_name}_calibration_corruption_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


class CalibrationCorrectionBenchmark:
    """Benchmark for evaluating VLM's ability to correct calibration errors."""
    
    def __init__(self, dataset_path: str, output_dir: str = "./calibration_benchmark_results", corruption_rate: float = 0.5):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.corruption_rate = corruption_rate
        
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
        """Run the calibration correction benchmark."""
        print("\n" + "=" * 60)
        print("CAMERA CALIBRATION CORRECTION BENCHMARK")
        print(f"Corruption Rate: {self.corruption_rate}")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset(max_trajectories)
        
        # Process trajectories
        process_fn = partial(
            process_single_trajectory_with_corruption, 
            output_dir=self.output_dir,
            corruption_rate=self.corruption_rate
        )
        results_dataset = dataset.map(process_fn).materialize()
        results = list(results_dataset.iter_rows())
        
        # Aggregate results
        total_trajectories = len(results)
        trajectories_with_calibration = 0
        total_cameras = 0
        corrupted_cameras = 0
        vlm_evaluations = 0
        high_confidence_evaluations = 0
        
        rotation_errors = []
        translation_errors = []
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in results:
            if result["has_calibration"]:
                trajectories_with_calibration += 1
                
                cameras_corrupted = 0
                for camera_name, camera_eval in result["camera_evaluations"].items():
                    total_cameras += 1
                    
                    if camera_eval["was_corrupted"]:
                        corrupted_cameras += 1
                        cameras_corrupted += 1
                        
                        if camera_eval["vlm_evaluation"] and camera_eval["vlm_evaluation"].get("confidence") != "UNKNOWN":
                            vlm_evaluations += 1
                            
                            if camera_eval["vlm_evaluation"]["confidence"] == "HIGH":
                                high_confidence_evaluations += 1
                            
                            # Collect accuracy metrics
                            if "vlm_rotation_error" in camera_eval:
                                rotation_errors.append(camera_eval["vlm_rotation_error"])
                            if "vlm_translation_error" in camera_eval:
                                translation_errors.append(camera_eval["vlm_translation_error"])
                
                status = "🔧" if cameras_corrupted > 0 else "✅"
                print(f"{status} {result['trajectory_name']}: {cameras_corrupted} cameras corrupted")
        
        # Calculate metrics
        actual_corruption_rate = corrupted_cameras / total_cameras if total_cameras > 0 else 0
        vlm_evaluation_rate = vlm_evaluations / corrupted_cameras if corrupted_cameras > 0 else 0
        high_confidence_rate = high_confidence_evaluations / vlm_evaluations if vlm_evaluations > 0 else 0
        
        mean_rotation_error = np.mean(rotation_errors) if rotation_errors else 0
        mean_translation_error = np.mean(translation_errors) if translation_errors else 0
        
        print(f"\nBenchmark Summary:")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Trajectories with calibration: {trajectories_with_calibration}")
        print(f"Total cameras evaluated: {total_cameras}")
        print(f"Cameras corrupted: {corrupted_cameras} ({actual_corruption_rate:.1%})")
        print(f"VLM evaluations completed: {vlm_evaluations} ({vlm_evaluation_rate:.1%} of corrupted)")
        print(f"High confidence evaluations: {high_confidence_evaluations} ({high_confidence_rate:.1%})")
        
        if rotation_errors:
            print(f"\nVLM Accuracy Metrics:")
            print(f"Mean rotation angle error: {mean_rotation_error:.2f}°")
            print(f"Mean translation magnitude error: {mean_translation_error:.3f}m")
        
        # Save summary
        summary = {
            "total_trajectories": total_trajectories,
            "trajectories_with_calibration": trajectories_with_calibration,
            "total_cameras": total_cameras,
            "corrupted_cameras": corrupted_cameras,
            "actual_corruption_rate": actual_corruption_rate,
            "vlm_evaluations": vlm_evaluations,
            "vlm_evaluation_rate": vlm_evaluation_rate,
            "high_confidence_evaluations": high_confidence_evaluations,
            "high_confidence_rate": high_confidence_rate,
            "mean_rotation_error_deg": mean_rotation_error,
            "mean_translation_error_m": mean_translation_error,
            "rotation_errors": rotation_errors,
            "translation_errors": translation_errors
        }
        
        summary_file = self.output_dir / "calibration_correction_benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {self.output_dir}/")
        
        return summary


def main():
    """Main function to run the calibration correction benchmark."""
    parser = argparse.ArgumentParser(description="Run camera calibration correction benchmark using VLM")
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
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.5,
        help="Rate at which to corrupt camera calibrations (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create and run benchmark
        benchmark = CalibrationCorrectionBenchmark(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            corruption_rate=args.corruption_rate
        )
        
        summary = benchmark.run_benchmark(max_trajectories=args.max_trajectories)
        
        print(f"\nFinal VLM Evaluation Rate: {summary['vlm_evaluation_rate']:.1%}")
        print(f"Mean Rotation Error: {summary['mean_rotation_error_deg']:.2f}°")
        print(f"Mean Translation Error: {summary['mean_translation_error_m']:.3f}m")
        
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()