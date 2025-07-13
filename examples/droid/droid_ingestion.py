"""
DROID Dataset Ingestion - Converts downloaded DROID data into RoboDM format.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import h5py
import cv2
import glob
import ray

import robodm
from robodm import Trajectory

# Camera names from DROID dataset
CAMERA_NAMES = ["wrist", "exterior_image_1", "exterior_image_2"]


def flatten_dict(data, parent_key='', sep='/'):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_hf_camera_extrinsics(cache_dir: Path):
    """Load camera extrinsics from cached HuggingFace files."""
    hf_extrinsics = {}
    
    json_files = {
        "cam2base_extrinsics": "cam2base_extrinsics.json",
        "cam2cam_extrinsics": "cam2cam_extrinsics.json",
        "cam2base_extrinsic_superset": "cam2base_extrinsic_superset.json"
    }
    
    for file_key, filename in json_files.items():
        cache_path = cache_dir / filename
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    hf_extrinsics[file_key] = json.load(f)
                print(f"Loaded {file_key} with {len(hf_extrinsics[file_key])} entries.")
            except Exception as e:
                print(f"Error loading {file_key}: {e}")
    
    return hf_extrinsics


def load_camera_intrinsics(download_dir: Path):
    """Load camera intrinsics from download directory."""
    intrinsics_path = download_dir / "camera_intrinsics_all.json"
    if intrinsics_path.exists():
        with open(intrinsics_path, 'r') as f:
            return json.load(f)
    return {}


def get_hf_camera_extrinsics(hf_extrinsics, episode_id, camera_serial):
    """Get camera extrinsics from HF data for a specific episode and camera."""
    # Try each source in order of preference
    for source in ["cam2base_extrinsic_superset", "cam2base_extrinsics", "cam2cam_extrinsics"]:
        if source in hf_extrinsics and hf_extrinsics[source]:
            if episode_id in hf_extrinsics[source]:
                entry = hf_extrinsics[source][episode_id]
                if str(camera_serial) in entry:
                    return entry[str(camera_serial)]
    return None


def load_mp4_frames(mp4_path: str) -> np.ndarray:
    """Load all frames from an MP4 file."""
    if not os.path.exists(mp4_path):
        return np.array([])
        
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return np.array(frames)


def split_stereo_frames(stereo_frames: np.ndarray):
    """Split side-by-side stereo frames into left and right."""
    if len(stereo_frames) == 0:
        return np.array([]), np.array([])
        
    num_frames, height, width, channels = stereo_frames.shape
    half_width = width // 2
    
    left_frames = stereo_frames[:, :, :half_width, :]
    right_frames = stereo_frames[:, :, half_width:, :]
    
    return left_frames, right_frames


@ray.remote
def process_episode(episode_dir: Path, output_dir: Path, hf_extrinsics: Dict, camera_intrinsics: Dict):
    """
    Process a single downloaded episode and convert to RoboDM format.
    """
    try:
        episode_id = episode_dir.name
        
        # Load download metadata
        download_metadata_path = episode_dir / "download_metadata.json"
        if not download_metadata_path.exists():
            print(f"No download metadata found for {episode_id}")
            return None
            
        with open(download_metadata_path, 'r') as f:
            download_metadata = json.load(f)
        
        if not download_metadata.get("download_success", False):
            print(f"Episode {episode_id} was not downloaded successfully, skipping")
            return None
        
        # Load TFDS data
        tfds_path = episode_dir / "tfds_data.json"
        if not tfds_path.exists():
            print(f"No TFDS data found for {episode_id}")
            return None
            
        with open(tfds_path, 'r') as f:
            tfds_data = json.load(f)
        
        steps_data = tfds_data.get("steps", [])
        if not steps_data:
            print(f"No steps data for {episode_id}")
            return None
        
        # Find raw data directory
        raw_data_dirs = list((episode_dir / "raw_data").glob("*"))
        if not raw_data_dirs:
            print(f"No raw data directory found for {episode_id}")
            return None
        
        scene_path = raw_data_dirs[0]
        
        # Load metadata JSON from raw data
        metadata = None
        json_files = glob.glob(str(scene_path) + "/*.json")
        if json_files:
            with open(json_files[0], "r") as f:
                metadata = json.load(f)
        
        # Get camera serials and create reverse mapping
        camera_serials = {}
        serial_to_camera_name = {}
        if metadata:
            # Map metadata keys to our camera names
            camera_key_mapping = {
                'wrist': 'wrist_cam_serial',
                'exterior_image_1': 'ext1_cam_serial',
                'exterior_image_2': 'ext2_cam_serial'
            }
            
            # First try the mapped keys
            for camera_name, serial_key in camera_key_mapping.items():
                if serial_key in metadata:
                    serial = metadata[serial_key]
                    camera_serials[camera_name] = serial
                    serial_to_camera_name[str(serial)] = camera_name
            
            # Also check for alternative key formats
            for key, value in metadata.items():
                if 'serial' in key.lower() and isinstance(value, (str, int)):
                    for camera_name in CAMERA_NAMES:
                        if camera_name in key:
                            if camera_name not in camera_serials:
                                camera_serials[camera_name] = str(value)
                                serial_to_camera_name[str(value)] = camera_name
        
        # Load trajectory H5 file
        h5_file = scene_path / "trajectory.h5"
        trajectory_data = {}
        traj_length = 0
        
        if not h5_file.exists():
            print(f"No trajectory.h5 file found for {episode_id}")
            return None
            
        with h5py.File(str(h5_file), "r") as f:
            # Get trajectory length
            if "action" in f:
                for key in f["action"].keys():
                    if isinstance(f["action"][key], h5py.Dataset):
                        traj_length = f["action"][key].shape[0]
                        break
            
            # Extract all data from H5 file
            def extract_h5_data(group, prefix=""):
                data = {}
                for key in group.keys():
                    full_key = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], h5py.Group):
                        data.update(extract_h5_data(group[key], full_key))
                    elif isinstance(group[key], h5py.Dataset):
                        # Read entire dataset into memory
                        data[full_key] = np.array(group[key])
                return data
            
            trajectory_data = extract_h5_data(f)
        
        # Find all unique camera serials in the H5 data and infer mappings
        h5_camera_serials = set()
        for key in trajectory_data.keys():
            if "observation/camera_extrinsics/" in key:
                parts = key.split('/')
                for i, part in enumerate(parts):
                    if part == "camera_extrinsics" and i + 1 < len(parts):
                        serial_side = parts[i + 1]
                        serial = serial_side.split('_')[0]
                        if serial.isdigit():
                            h5_camera_serials.add(serial)
        
        # Infer camera mappings for unmapped serials
        if h5_camera_serials:
            unmapped_serials = h5_camera_serials - set(serial_to_camera_name.keys())
            if unmapped_serials:
                unmapped_list = sorted(list(unmapped_serials))
                missing_cameras = [cam for cam in CAMERA_NAMES if cam not in camera_serials]
                
                # If we have exactly 2 unmapped serials and 2 missing exterior cameras
                if len(unmapped_list) == 2 and 'exterior_image_1' in missing_cameras and 'exterior_image_2' in missing_cameras:
                    serial_to_camera_name[unmapped_list[0]] = 'exterior_image_1'
                    serial_to_camera_name[unmapped_list[1]] = 'exterior_image_2'
                    camera_serials['exterior_image_1'] = unmapped_list[0]
                    camera_serials['exterior_image_2'] = unmapped_list[1]
        
        # Rename camera extrinsics keys from serial numbers to camera names
        renamed_trajectory_data = {}
        for key, data in trajectory_data.items():
            new_key = key
            if "observation/camera_extrinsics/" in key:
                parts = key.split('/')
                for i, part in enumerate(parts):
                    if part == "camera_extrinsics" and i + 1 < len(parts):
                        serial_side = parts[i + 1]
                        serial_parts = serial_side.split('_')
                        if len(serial_parts) >= 1:
                            serial = serial_parts[0]
                            side_suffix = '_'.join(serial_parts[1:]) if len(serial_parts) > 1 else ''
                            if serial in serial_to_camera_name:
                                camera_name = serial_to_camera_name[serial]
                                parts[i + 1] = f"{camera_name}_{side_suffix}" if side_suffix else camera_name
                                new_key = '/'.join(parts)
                        break
            renamed_trajectory_data[new_key] = data
        trajectory_data = renamed_trajectory_data
        
        # Load camera images
        camera_frames = {}
        recordings_path = scene_path / "recordings" / "MP4"
        
        if recordings_path.exists() and metadata:
            # Map camera names to MP4 files
            mp4_mappings = {
                "wrist": metadata.get("wrist_mp4_path", ""),
                "exterior_image_1": metadata.get("ext1_mp4_path", ""),
                "exterior_image_2": metadata.get("ext2_mp4_path", "")
            }
            
            for camera_name, mp4_path in mp4_mappings.items():
                if mp4_path:
                    mp4_filename = os.path.basename(mp4_path)
                    full_mp4_path = recordings_path / mp4_filename
                    
                    # Try stereo version first
                    stereo_filename = mp4_filename.replace(".mp4", "-stereo.mp4")
                    stereo_path = recordings_path / stereo_filename
                    
                    if stereo_path.exists():
                        print(f"Loading stereo frames for {camera_name}")
                        stereo_frames = load_mp4_frames(str(stereo_path))
                        if len(stereo_frames) > 0:
                            left_frames, right_frames = split_stereo_frames(stereo_frames)
                            camera_frames[f"{camera_name}_left"] = left_frames
                            camera_frames[f"{camera_name}_right"] = right_frames
                    elif full_mp4_path.exists():
                        print(f"Loading frames for {camera_name}")
                        frames = load_mp4_frames(str(full_mp4_path))
                        if len(frames) > 0:
                            camera_frames[f"{camera_name}_left"] = frames
        
        # Verify we have valid trajectory data
        if traj_length == 0:
            print(f"Skipping {episode_id} - no trajectory data in H5 file")
            return None
        
        # Create output RoboDM trajectory
        output_path = output_dir / f"{episode_id}.vla"
        traj = robodm.Trajectory(path=str(output_path), mode="w")
        
        # Process each timestep
        for t in range(traj_length):
            # Add TFDS data
            if t < len(steps_data):
                step = steps_data[t]
                # Flatten and add all TFDS data
                flat_tfds = flatten_dict(step)
                for key, value in flat_tfds.items():
                    # Convert lists back to numpy arrays
                    if isinstance(value, list):
                        traj.add(f"tfds/{key}", np.array(value))
                    else:
                        traj.add(f"tfds/{key}", value)
            
            # Add raw trajectory data from H5
            for key, data in trajectory_data.items():
                if isinstance(data, np.ndarray) and len(data.shape) > 0 and t < data.shape[0]:
                    value = data[t]
                    traj.add(f"raw/h5/{key}", value)
            
            # Add camera intrinsics and extrinsics
            for camera_name, serial in camera_serials.items():
                # Try to get HF extrinsics first
                hf_extrinsic = get_hf_camera_extrinsics(hf_extrinsics, episode_id, serial)
                if hf_extrinsic:
                    traj.add(f"raw/camera_extrinsics/{camera_name}/hf", np.array(hf_extrinsic))
                
                # Add extrinsics from metadata if available
                extrinsic_key_mapping = {
                    'wrist': 'wrist_cam_extrinsics',
                    'exterior_image_1': 'ext1_cam_extrinsics',
                    'exterior_image_2': 'ext2_cam_extrinsics'
                }
                
                if metadata and camera_name in extrinsic_key_mapping:
                    metadata_key = extrinsic_key_mapping[camera_name]
                    if metadata_key in metadata:
                        extrinsic_data = metadata[metadata_key]
                        traj.add(f"raw/camera_extrinsics/{camera_name}/left", np.array(extrinsic_data))
                
                # Also add any extrinsics from the H5 file
                for side in ["left", "right"]:
                    extrinsic_key = f"observation/camera_extrinsics/{camera_name}_{side}"
                    if extrinsic_key in trajectory_data:
                        data = trajectory_data[extrinsic_key]
                        if isinstance(data, np.ndarray) and len(data.shape) > 0 and t < data.shape[0]:
                            value = data[t]
                            traj.add(f"raw/camera_extrinsics/{camera_name}/{side}", value)
                
                # Add camera intrinsics if available
                if serial in camera_intrinsics:
                    intrinsic_data = camera_intrinsics[serial]
                    intrinsic_matrix = np.array(intrinsic_data['intrinsic_matrix'])
                    traj.add(f"raw/camera_intrinsics/{camera_name}", intrinsic_matrix)
            
            # Add image data
            for cam_key, frames in camera_frames.items():
                if t < len(frames):
                    traj.add(f"raw/images/{cam_key}", frames[t])
        
        # Determine task success from path
        gs_path = download_metadata.get("gs_path", "")
        task_successful = 'success' in gs_path.lower()
        
        # Add metadata
        metadata_dict = {
            "episode_id": episode_id,
            "language_instruction": tfds_data.get("language_instruction", ""),
            "trajectory_length": traj_length,
            "task_successful": task_successful,
            "gsutil_path": gs_path,
            "camera_serials": camera_serials,
            "tfds_file_path": download_metadata.get("tfds_file_path", "")
        }
        
        # Store metadata as a string
        metadata_str = json.dumps(metadata_dict)
        traj.add("metadata", metadata_str)
        
        # Close trajectory
        traj.close()
        
        print(f"Successfully processed {episode_id} -> {output_path}")
        return str(output_path)
        
    except Exception as e:
        import traceback
        print(f"Error processing episode {episode_id}: {e}")
        traceback.print_exc()
        return None


def ingest_droid_from_downloads(
    download_dir: str = "./droid_downloaded_data",
    output_dir: str = "./droid_combined_data",
    num_workers: int = 64
):
    """
    Ingest DROID dataset from downloaded data.
    
    Args:
        download_dir: Directory containing downloaded data
        output_dir: Directory to save RoboDM trajectories
        num_workers: Number of parallel workers
    """
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    # Create output directory
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HuggingFace camera extrinsics
    print("Loading HuggingFace camera extrinsics...")
    hf_cache_dir = download_dir / "huggingface_cache"
    hf_extrinsics = load_hf_camera_extrinsics(hf_cache_dir)
    
    # Load camera intrinsics
    print("Loading camera intrinsics...")
    camera_intrinsics = load_camera_intrinsics(download_dir)
    if camera_intrinsics:
        print(f"Loaded intrinsics for {len(camera_intrinsics)} camera serials")
    
    # Find all episode directories
    episode_dirs = [d for d in download_dir.iterdir() 
                    if d.is_dir() and d.name != "huggingface_cache"]
    
    print(f"Found {len(episode_dirs)} episode directories to process")
    
    # Process episodes in parallel
    futures = []
    for episode_dir in episode_dirs:
        future = process_episode.remote(episode_dir, output_dir, hf_extrinsics, camera_intrinsics)
        futures.append(future)
        
        # Limit concurrent tasks
        if len(futures) >= num_workers:
            ready, futures = ray.wait(futures, num_returns=1)
            for f in ready:
                result = ray.get(f)
                if result:
                    print(f"Completed: {result}")
    
    # Wait for remaining tasks
    results = ray.get(futures)
    successful = [r for r in results if r is not None]
    
    print(f"\nIngestion complete!")
    print(f"Successfully processed {len(successful)} out of {len(episode_dirs)} episodes")
    print(f"Output directory: {output_dir}")
    
    # Create a RoboDM dataset from the saved trajectories
    from robodm.dataset import VLADataset
    dataset = VLADataset(str(output_dir / "*.vla"))
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", default="./droid_downloaded_data",
                        help="Directory containing downloaded data")
    parser.add_argument("--output_dir", default="./droid_combined_data",
                        help="Directory to save RoboDM trajectories")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    dataset = ingest_droid_from_downloads(
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    print(f"\nCreated dataset with {dataset.count()} trajectories")