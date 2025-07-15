"""
DROID Dataset Downloader - Downloads TFDS and raw trajectory data to local directories.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import ray
import json
import numpy as np
import requests
import shutil
import csv

# URLs to the camera extrinsics JSON files on Hugging Face
HF_JSON_URLS = {
    "cam2base_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsics.json",
    "cam2cam_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2cam_extrinsics.json",
    "cam2base_extrinsic_superset": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsic_superset.json"
}


def download_hf_camera_extrinsics(cache_dir: Path):
    """Download camera extrinsics from HuggingFace."""
    cache_dir.mkdir(exist_ok=True)
    
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
            except Exception as e:
                print(f"Error downloading {file_key}: {e}")


def extract_camera_intrinsics_with_zed(recordings_path: Path, camera_serials: List[str]) -> dict:
    """Extract camera intrinsics using ZED SDK for each camera serial."""
    camera_intrinsics = {}
    
    for serial in camera_serials:
        try:
            import pyzed.sl as sl
            init_params = sl.InitParameters()
            svo_path = recordings_path / "SVO" / f"{serial}.svo"
            
            if not svo_path.exists():
                print(f"SVO file not found for camera {serial}: {svo_path}")
                continue
                
            init_params.set_from_svo_file(str(svo_path))
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = 0.2

            zed = sl.Camera()
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise Exception(f"Error reading camera data: {err}")

            params = zed.get_camera_information().camera_configuration.calibration_parameters
            
            left_intrinsic_mat = [
                [params.left_cam.fx, 0, params.left_cam.cx],
                [0, params.left_cam.fy, params.left_cam.cy],
                [0, 0, 1],
            ]
            right_intrinsic_mat = [
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ]
            
            camera_intrinsics[serial] = {
                'left_intrinsic_matrix': left_intrinsic_mat,
                'right_intrinsic_matrix': right_intrinsic_mat,
                'left_fx': params.left_cam.fx,
                'left_fy': params.left_cam.fy,
                'left_cx': params.left_cam.cx,
                'left_cy': params.left_cam.cy,
                'right_fx': params.right_cam.fx,
                'right_fy': params.right_cam.fy,
                'right_cx': params.right_cam.cx,
                'right_cy': params.right_cam.cy
            }
            
            zed.close()
            print(f"Successfully extracted intrinsics for camera {serial} using ZED SDK")
            
        except (ModuleNotFoundError, Exception) as e:
            print(f"ZED SDK not available or error for camera {serial}: {e}")
            # Use default intrinsics as fallback
            default_intrinsic_mat = [
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ]
            camera_intrinsics[serial] = {
                'left_intrinsic_matrix': default_intrinsic_mat,
                'right_intrinsic_matrix': default_intrinsic_mat,
                'left_fx': 733.37261963,
                'left_fy': 733.37261963,
                'left_cx': 625.26251221,
                'left_cy': 361.92279053,
                'right_fx': 733.37261963,
                'right_fy': 733.37261963,
                'right_cx': 625.26251221,
                'right_cy': 361.92279053,
                'is_default': True
            }
    
    return camera_intrinsics


def extract_camera_intrinsics_from_metadata(metadata: dict) -> dict:
    """Extract camera intrinsics from episode metadata and format as 3x3 matrices."""
    camera_intrinsics = {}
    
    # Camera intrinsic keys mapping
    intrinsic_keys = {
        'wrist': {
            'serial': 'wrist_cam_serial',
            'fx': 'wrist_cam_fx',
            'fy': 'wrist_cam_fy',
            'cx': 'wrist_cam_cx',
            'cy': 'wrist_cam_cy'
        },
        'exterior_image_1': {
            'serial': 'ext1_cam_serial',
            'fx': 'ext1_cam_fx',
            'fy': 'ext1_cam_fy',
            'cx': 'ext1_cam_cx',
            'cy': 'ext1_cam_cy'
        },
        'exterior_image_2': {
            'serial': 'ext2_cam_serial',
            'fx': 'ext2_cam_fx',
            'fy': 'ext2_cam_fy',
            'cx': 'ext2_cam_cx',
            'cy': 'ext2_cam_cy'
        }
    }
    
    # Extract intrinsics for each camera
    for camera_name, keys in intrinsic_keys.items():
        if keys['serial'] in metadata:
            serial = str(metadata[keys['serial']])
            
            # Check if all intrinsic parameters exist
            if all(keys[param] in metadata for param in ['fx', 'fy', 'cx', 'cy']):
                # Create 3x3 intrinsic matrix
                intrinsic_matrix = [
                    [metadata[keys['fx']], 0, metadata[keys['cx']]],
                    [0, metadata[keys['fy']], metadata[keys['cy']]],
                    [0, 0, 1]
                ]
                camera_intrinsics[serial] = {
                    'camera_name': camera_name,
                    'intrinsic_matrix': intrinsic_matrix,
                    'fx': metadata[keys['fx']],
                    'fy': metadata[keys['fy']],
                    'cx': metadata[keys['cx']],
                    'cy': metadata[keys['cy']]
                }
    
    return camera_intrinsics


def convert_to_serializable(obj):
    """Recursively convert numpy arrays and other non-serializable types to serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, 'numpy'):
        # Handle TensorFlow tensors
        return convert_to_serializable(obj.numpy())
    else:
        return obj


def extract_episode_metadata(episode, episode_idx: int) -> dict:
    """
    Extract episode metadata from TFDS (runs in main process).
    
    Returns:
        dict: Episode metadata including ID and file path
    """
    # Extract episode ID from file path
    file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
    episode_id_match = re.search(r'([^/]+)/trajectory\.h5$', file_path)
    episode_id = episode_id_match.group(1) if episode_id_match else f"episode_{episode_idx}"
    
    # Extract language instruction
    steps = list(episode["steps"].as_numpy_iterator())
    language_instruction = steps[0]["language_instruction"].decode("utf-8") if steps else ""
    
    return {
        "episode_id": episode_id,
        "file_path": file_path,
        "language_instruction": language_instruction
    }


@ray.remote(num_gpus=0.01)
def download_raw_data_and_extract_intrinsics(episode_metadata: dict, output_dir: Path):
    """
    Download raw data and extract camera intrinsics using ZED (runs in Ray).
    
    Args:
        episode_metadata: Dict containing episode_id, file_path, and language_instruction
        output_dir: Base output directory
        
    Returns:
        dict: Download status and camera intrinsics info
    """
    try:
        episode_id = episode_metadata["episode_id"]
        file_path = episode_metadata["file_path"]
        episode_output_dir = output_dir / episode_id
        episode_output_dir.mkdir(parents=True, exist_ok=True)
        
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
        raw_data_dir = episode_output_dir / "raw_data"
        
        # Download raw data
        try:
            # Create the raw_data directory first
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Use gsutil to copy the contents of the episode folder
            # Remove trailing slash from gs_path and copy contents to raw_data_dir
            gs_path_clean = gs_path.rstrip('/')
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{gs_path_clean}/*", str(raw_data_dir) + "/"],
                capture_output=True,
                check=True
            )
            print(f"Downloaded raw data for {episode_id}")
            
            # Find and load metadata JSON from raw data to get camera serials
            camera_intrinsics = {}
            camera_serials = []
            
            # Look for JSON files directly in raw_data_dir
            json_files = list(raw_data_dir.glob("*.json"))
            print(f"Found JSON files: {json_files}")
            
            if json_files:
                # Load the first metadata JSON file
                with open(json_files[0], 'r') as f:
                    raw_metadata = json.load(f)
                    
                    # Extract camera serials from metadata
                    serial_keys = ['wrist_cam_serial', 'ext1_cam_serial', 'ext2_cam_serial']
                    for key in serial_keys:
                        if key in raw_metadata:
                            camera_serials.append(str(raw_metadata[key]))
                    
                    print("camera_serial", camera_serials)
                    # Try to extract intrinsics using ZED SDK first
                    if camera_serials:
                        recordings_path = raw_data_dir / "recordings"
                        camera_intrinsics = extract_camera_intrinsics_with_zed(recordings_path, camera_serials)
                    
                    # If ZED SDK extraction failed or incomplete, fall back to metadata
                    if not camera_intrinsics:
                        camera_intrinsics = extract_camera_intrinsics_from_metadata(raw_metadata)
                    
                    if camera_intrinsics:
                        # Save camera intrinsics to separate file
                        intrinsics_path = episode_output_dir / "camera_intrinsics.json"
                        with open(intrinsics_path, 'w') as f:
                            json.dump(camera_intrinsics, f, indent=2)
                        print(f"Saved camera intrinsics for {episode_id}")
            
            # Save download metadata
            metadata = {
                "episode_id": episode_id,
                "tfds_file_path": file_path,
                "gs_path": gs_path,
                "download_success": True,
                "has_camera_intrinsics": bool(camera_intrinsics)
            }
            
        except Exception as e:
            print(f"Failed to download raw data for {episode_id}: {e}")
            metadata = {
                "episode_id": episode_id,
                "tfds_file_path": file_path,
                "gs_path": gs_path,
                "download_success": False,
                "error": str(e)
            }
        
        # Save download metadata
        metadata_path = episode_output_dir / "download_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
        
    except Exception as e:
        import traceback
        print(f"Error processing episode {episode_metadata.get('episode_id', 'unknown')}: {e}")
        traceback.print_exc()
        
        return {
            "episode_id": episode_metadata.get("episode_id", "unknown"),
            "download_success": False,
            "error": str(e)
        }


def download_droid_dataset(
    output_dir: str = "./droid_downloaded_data",
    num_episodes: int = 10,
    num_workers: int = 64
):
    """
    Download DROID dataset from TFDS and raw sources.
    TFDS data is saved directly in the main process to avoid passing large data through Ray.
    Ray is used only for downloading raw data and extracting camera intrinsics with ZED.
    Creates a CSV file with episode metadata for ingestion.
    
    Args:
        output_dir: Directory to save downloaded data
        num_episodes: Number of episodes to download
        num_workers: Number of parallel workers for raw data download
    """
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download HuggingFace camera extrinsics
    print("Downloading HuggingFace camera extrinsics...")
    hf_cache_dir = output_dir / "huggingface_cache"
    download_hf_camera_extrinsics(hf_cache_dir)
    
    try:
        # Load TFDS dataset
        print("Loading DROID dataset from TFDS...")
        # ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")
        ds = tfds.load("droid_100", data_dir="/root/droid-example", split="train")
        
        # First pass: Extract episode metadata from TFDS (no Ray)
        print("Extracting episode metadata from TFDS...")
        episode_metadata_list = []
        for i, episode in enumerate(ds.take(num_episodes)):
            metadata = extract_episode_metadata(episode, i)
            episode_metadata_list.append(metadata)
        
        # Second pass: Download raw data and extract intrinsics using Ray
        print("Downloading raw data and extracting camera intrinsics...")
        futures = []
        for metadata in episode_metadata_list:
            future = download_raw_data_and_extract_intrinsics.remote(metadata, output_dir)
            futures.append(future)
            
            # Limit concurrent tasks
            if len(futures) >= num_workers:
                ready, futures = ray.wait(futures, num_returns=1)
                for f in ready:
                    result = ray.get(f)
                    if result:
                        print(f"Completed raw data download: {result.get('episode_id', 'unknown')}")
        
        # Wait for remaining tasks
        results = ray.get(futures)
        successful = [r for r in results if r and r.get("download_success", False)]
        
        print(f"\nDownload complete!")
        print(f"Successfully downloaded {len(successful)} out of {num_episodes} episodes")
        print(f"Output directory: {output_dir}")
        
        # Aggregate all camera intrinsics
        all_camera_intrinsics = {}
        intrinsics_by_episode = {}
        
        for episode_dir in output_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name != "huggingface_cache":
                intrinsics_path = episode_dir / "camera_intrinsics.json"
                if intrinsics_path.exists():
                    with open(intrinsics_path, 'r') as f:
                        episode_intrinsics = json.load(f)
                    
                    # Add to episode mapping
                    intrinsics_by_episode[episode_dir.name] = episode_intrinsics
                    
                    # Add to global mapping (serial -> intrinsics)
                    for serial, intrinsics_data in episode_intrinsics.items():
                        if serial not in all_camera_intrinsics:
                            all_camera_intrinsics[serial] = intrinsics_data
        
        # Save aggregated camera intrinsics
        if all_camera_intrinsics:
            global_intrinsics_path = output_dir / "camera_intrinsics_all.json"
            with open(global_intrinsics_path, 'w') as f:
                json.dump(all_camera_intrinsics, f, indent=2)
            print(f"Saved global camera intrinsics mapping to: {global_intrinsics_path}")
            
            # Also save episode-to-intrinsics mapping
            episode_intrinsics_path = output_dir / "camera_intrinsics_by_episode.json"
            with open(episode_intrinsics_path, 'w') as f:
                json.dump(intrinsics_by_episode, f, indent=2)
        
        # Create summary file
        summary = {
            "total_episodes": num_episodes,
            "successful_downloads": len(successful),
            "failed_downloads": num_episodes - len(successful),
            "episodes": results,
            "total_camera_serials_with_intrinsics": len(all_camera_intrinsics)
        }
        
        summary_path = output_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Download summary saved to: {summary_path}")
        
        # Create CSV file with episode metadata
        csv_path = output_dir / "episode_metadata.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'episode_id', 
                'raw_data_path', 
                'tfds_file_path',
                'language_instruction',
                'wrist_serial', 
                'wrist_intrinsics',
                'wrist_extrinsics',
                'ext1_serial', 
                'ext1_intrinsics',
                'ext1_extrinsics',
                'ext2_serial',
                'ext2_intrinsics',
                'ext2_extrinsics',
                'task_successful'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Combine episode metadata with download results
            episode_map = {m["episode_id"]: m for m in episode_metadata_list}
            
            # Process each episode directory
            for episode_dir in sorted(output_dir.iterdir()):
                if episode_dir.is_dir() and episode_dir.name != "huggingface_cache":
                    episode_id = episode_dir.name
                    row_data = {'episode_id': episode_id}
                    
                    # Add TFDS metadata
                    if episode_id in episode_map:
                        tfds_meta = episode_map[episode_id]
                        row_data['tfds_file_path'] = tfds_meta['file_path']
                        row_data['language_instruction'] = tfds_meta['language_instruction']
                    
                    # Check if download was successful
                    download_metadata_path = episode_dir / "download_metadata.json"
                    if download_metadata_path.exists():
                        with open(download_metadata_path, 'r') as f:
                            download_meta = json.load(f)
                        
                        if not download_meta.get("download_success", False):
                            continue
                        
                        # Task success from GS path
                        gs_path = download_meta.get("gs_path", "")
                        row_data['task_successful'] = 'success' in gs_path.lower()
                    
                    # Find raw data path - should be the raw_data directory itself
                    raw_data_path = episode_dir / "raw_data"
                    if raw_data_path.exists():
                        row_data['raw_data_path'] = str(raw_data_path)
                    
                    # Load camera intrinsics
                    intrinsics_path = episode_dir / "camera_intrinsics.json"
                    if intrinsics_path.exists():
                        with open(intrinsics_path, 'r') as f:
                            episode_intrinsics = json.load(f)
                        
                        # Process each camera serial
                        for serial, intrinsics_data in episode_intrinsics.items():
                            camera_name = intrinsics_data.get('camera_name', '')
                            
                            if camera_name == 'wrist':
                                row_data['wrist_serial'] = serial
                                row_data['wrist_intrinsics'] = json.dumps(intrinsics_data.get('intrinsic_matrix', []))
                            elif camera_name == 'exterior_image_1':
                                row_data['ext1_serial'] = serial
                                row_data['ext1_intrinsics'] = json.dumps(intrinsics_data.get('intrinsic_matrix', []))
                            elif camera_name == 'exterior_image_2':
                                row_data['ext2_serial'] = serial
                                row_data['ext2_intrinsics'] = json.dumps(intrinsics_data.get('intrinsic_matrix', []))
                                        
                    writer.writerow(row_data)
        
        print(f"Episode metadata CSV saved to: {csv_path}")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./droid_downloaded_data", 
                        help="Directory to save downloaded data")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to download")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    
    download_droid_dataset(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        num_workers=args.num_workers
    )