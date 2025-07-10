"""
Simple DROID ingestion pipeline that combines TFDS and raw trajectory data.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, List
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import ray
import json
import numpy as np
import h5py
import cv2
import glob
import requests

import robodm
from robodm import Trajectory

# Camera names from DROID dataset
CAMERA_NAMES = ["wrist", "exterior_image_1", "exterior_image_2"]

# URLs to the camera extrinsics JSON files on Hugging Face
HF_JSON_URLS = {
    "cam2base_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsics.json",
    "cam2cam_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2cam_extrinsics.json",
    "cam2base_extrinsic_superset": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsic_superset.json"
}


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


def load_hf_camera_extrinsics():
    """Download and load camera extrinsics from HuggingFace."""
    cache_dir = Path("./huggingface_cache")
    cache_dir.mkdir(exist_ok=True)
    
    hf_extrinsics = {}
    
    for file_key, url in HF_JSON_URLS.items():
        cache_path = cache_dir / f"{file_key}.json"
        
        # Download if not cached
        if not cache_path.exists():
            try:
                print(f"Downloading {file_key} from Hugging Face...")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {file_key} successfully.")
                else:
                    print(f"Failed to download {file_key}: {response.status_code}")
                    continue
            except Exception as e:
                print(f"Error downloading {file_key}: {e}")
                continue
        
        # Load the JSON file
        try:
            with open(cache_path, 'r') as f:
                hf_extrinsics[file_key] = json.load(f)
            print(f"Loaded {file_key} with {len(hf_extrinsics[file_key])} entries.")
        except Exception as e:
            print(f"Error loading {file_key}: {e}")
    
    return hf_extrinsics


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
def process_episode_combined(episode, episode_idx: int, output_dir: str, temp_dir: str, hf_extrinsics: Dict):
    """
    Process a single TFDS episode by:
    1. Getting TFDS data
    2. Downloading raw trajectory
    3. Combining both into a single RoboDM trajectory
    """
    try:
        # Extract TFDS data
        tfds_data = episode  # Already pre-extracted
        
        # Extract episode ID from file path
        file_path = tfds_data["episode_metadata"]["file_path"]
        print(file_path)
        episode_id_match = re.search(r'([^/]+)/trajectory\.h5$', file_path)
        episode_id = episode_id_match.group(1) if episode_id_match else f"episode_{episode_idx}"
        
        # Process all steps from TFDS
        steps_data = []
        for step in tfds_data["steps"]:
            step_dict = {}
            
            # Extract all fields from the step
            for key, value in step.items():
                if isinstance(value, bytes):
                    step_dict[key] = value.decode("utf-8")
                elif hasattr(value, 'numpy'):
                    step_dict[key] = value.numpy()
                else:
                    step_dict[key] = value
            
            steps_data.append(step_dict)
        
        tfds_data["steps"] = steps_data
        tfds_data["language_instruction"] = steps_data[0]["language_instruction"] if steps_data else ""
        
        print(f"Processing episode {episode_id} with {len(steps_data)} steps")
        
        # Download raw trajectory
        path_parts = file_path.replace("/trajectory.h5", "").split('/')
        try:
            base_index = path_parts.index("droid_raw")
            if path_parts[base_index+1] != '1.0.1':
                raise ValueError("Found 'droid_raw' but not '1.0.1' following it.")
            episode_folder = "/".join(path_parts[base_index+2:])
        except (ValueError, IndexError):
            episode_folder = "/".join(path_parts[-4:])
        
        gs_path = f"gs://gresearch/robotics/droid_raw/1.0.1/{episode_folder}/"
        local_path = Path(temp_dir) / episode_id
        
        # Download raw data
        local_path.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", gs_path, str(local_path)],
                capture_output=True,
                check=True
            )
            
            # Find the actual downloaded directory
            downloaded_dirs = list(local_path.iterdir())
            if not downloaded_dirs:
                raise Exception("No data downloaded")
            scene_path = downloaded_dirs[0]
            
        except Exception as e:
            print(f"Failed to download raw data for {episode_id}: {e}")
            return None
        
        # Load metadata JSON
        metadata = None
        json_files = glob.glob(str(scene_path) + "/*.json")
        if json_files:
            with open(json_files[0], "r") as f:
                metadata = json.load(f)
        
        # Get camera serials
        camera_serials = {}
        if metadata:
            for camera_name in CAMERA_NAMES:
                serial_key = f"{camera_name}_cam_serial"
                if serial_key in metadata:
                    camera_serials[camera_name] = metadata[serial_key]
        
        # Load trajectory H5 file
        h5_file = scene_path / "trajectory.h5"
        trajectory_data = {}
        traj_length = 0
        
        if h5_file.exists():
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
                            # Store dataset reference for later extraction by timestep
                            data[full_key] = group[key]
                    return data
                
                # Extract and store all H5 data in memory before closing file
                trajectory_data_refs = extract_h5_data(f)
                
                # Convert H5 dataset references to actual numpy arrays
                trajectory_data = {}
                for key, dataset in trajectory_data_refs.items():
                    if isinstance(dataset, h5py.Dataset):
                        # Read entire dataset into memory
                        trajectory_data[key] = np.array(dataset)
                    else:
                        trajectory_data[key] = dataset
        
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
        
        # Create output RoboDM trajectory
        output_path = Path(output_dir) / f"{episode_id}.vla"
        traj = robodm.Trajectory(path=str(output_path), mode="w")
        
        # Process each timestep
        for t in range(traj_length):
            # Add TFDS data
            if t < len(steps_data):
                step = steps_data[t]
                # Flatten and add all TFDS data
                flat_tfds = flatten_dict(step)
                for key, value in flat_tfds.items():
                    # Handle numpy arrays
                    if isinstance(value, np.ndarray):
                        # Keep as numpy array for robodm
                        traj.add(f"tfds/{key}", value)
                    elif isinstance(value, (list, tuple)):
                        # Convert lists to numpy arrays
                        traj.add(f"tfds/{key}", np.array(value))
                    else:
                        # Scalar values
                        traj.add(f"tfds/{key}", value)
            
            # Add raw trajectory data from H5
            for key, data in trajectory_data.items():
                if isinstance(data, np.ndarray) and len(data.shape) > 0 and t < data.shape[0]:
                    value = data[t]
                    # Keep numpy arrays as is for robodm
                    traj.add(f"raw/h5/{key}", value)
            
            # Add camera intrinsics and extrinsics
            for camera_name, serial in camera_serials.items():
                # Try to get HF extrinsics first
                hf_extrinsic = get_hf_camera_extrinsics(hf_extrinsics, episode_id, serial)
                if hf_extrinsic:
                    traj.add(f"raw/camera_extrinsics/{camera_name}/hf", np.array(hf_extrinsic))
                
                # Also add any extrinsics from the H5 file
                for side in ["left", "right"]:
                    extrinsic_key = f"observation/camera_extrinsics/{serial}_{side}"
                    if extrinsic_key in trajectory_data:
                        data = trajectory_data[extrinsic_key]
                        if isinstance(data, np.ndarray) and len(data.shape) > 0 and t < data.shape[0]:
                            value = data[t]
                            traj.add(f"raw/camera_extrinsics/{camera_name}/{side}", value)
            
            # Add image data
            for cam_key, frames in camera_frames.items():
                if t < len(frames):
                    traj.add(f"raw/images/{cam_key}", frames[t])
        
        # Determine task success from path
        task_successful = 'success' in gs_path.lower()
        
        # Add metadata
        metadata_dict = {
            "episode_id": episode_id,
            "language_instruction": tfds_data["language_instruction"],
            "trajectory_length": traj_length,
            "task_successful": task_successful,
            "gsutil_path": gs_path,
            "camera_serials": camera_serials,
            "tfds_file_path": file_path
        }
        
        # Store metadata as a string (not numpy array)
        metadata_str = json.dumps(metadata_dict)
        # Store as a single-element string array to maintain compatibility
        traj.add("metadata", metadata_str)
        
        # Close trajectory
        traj.close()
        
        # Clean up downloaded files
        import shutil
        if scene_path.exists():
            shutil.rmtree(scene_path)
        
        print(f"Successfully processed {episode_id} -> {output_path}")
        return str(output_path)
        
    except Exception as e:
        import traceback
        print(f"Error processing episode {episode_idx}: {e}")
        traceback.print_exc()
        return None


def ingest_droid_combined(
    output_dir: str = "./droid_combined_data",
    num_episodes: int = 10,
    num_workers: int = 64
):
    """
    Ingest DROID dataset combining TFDS and raw trajectory data.
    
    Args:
        output_dir: Directory to save combined trajectories
        num_episodes: Number of episodes to process
        num_workers: Number of parallel workers
    """
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    # Load HuggingFace camera extrinsics
    print("Loading HuggingFace camera extrinsics...")
    hf_extrinsics = load_hf_camera_extrinsics()
    
    # Create directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="droid_combined_")
    
    try:
        # Load TFDS dataset
        print("Loading DROID dataset from TFDS...")
        ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")
        
        # Process episodes in parallel
        futures = []
        for i, episode in enumerate(ds.take(num_episodes)):
            # Extract data from TensorFlow dataset to make it serializable
            episode_data = {
                "episode_metadata": {
                    "file_path": episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
                },
                "steps": list(episode["steps"].as_numpy_iterator())
            }
            
            future = process_episode_combined.remote(
                episode_data, i, str(output_dir), temp_dir, hf_extrinsics
            )
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
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed {len(successful)} out of {num_episodes} episodes")
        print(f"Output directory: {output_dir}")
        
        # Create a RoboDM dataset from the saved trajectories
        from robodm.dataset import VLADataset
        dataset = VLADataset(str(output_dir / "*.vla"))
        
        return dataset
        
    finally:
        # Clean up temp directory
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./droid_combined_data")
    parser.add_argument("--num_episodes", type=int, default=500)
    
    args = parser.parse_args()
    

    # Just run the ingestion
    dataset = ingest_droid_combined(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes
    )
    print(f"\nCreated dataset with {dataset.count()} trajectories")