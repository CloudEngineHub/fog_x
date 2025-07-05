import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import ray

import robodm
from robodm import Trajectory


@ray.remote
def download_and_convert_trajectory(trajectory_path: str, output_dir: str, temp_dir: str) -> Tuple[bool, str, str]:
    """
    Download and convert a single DROID trajectory to RoboDM format.
    
    Args:
        trajectory_path: GCS path to DROID trajectory
        output_dir: Directory to save RoboDM trajectories
        temp_dir: Temporary directory for downloads
        
    Returns:
        Tuple of (success: bool, output_path: str, error_msg: str)
    """
    converter = DROIDProcessor()
    
    try:
        # Download trajectory
        traj_name = trajectory_path.rstrip("/").split("/")[-1]
        local_path = os.path.join(temp_dir, traj_name)
        
        # Download using gsutil
        parent_dir = os.path.dirname(local_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", trajectory_path, parent_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Load DROID data
        droid_data = converter.load_droid_trajectory(local_path)
        
        # Generate output filename
        success_or_failure = "success" if "success" in trajectory_path else "failure"
        output_path = os.path.join(output_dir, f"{success_or_failure}_{traj_name}.vla")
        
        # Convert to RoboDM
        converter.convert_to_robodm(droid_data, output_path)
        
        # Clean up downloaded files
        import shutil
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        return True, output_path, ""
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {trajectory_path}: {e}\n{traceback.format_exc()}"
        return False, "", error_msg


class DROIDProcessor:
    """Downloads and converts DROID trajectories to RoboDM format."""

    def __init__(self, base_path: str = "gs://gresearch/robotics/droid_raw/1.0.1/"):
        self.base_path = base_path
        self.camera_names = [
            "hand_camera_left_image",
            "hand_camera_right_image",
            "varied_camera_1_left_image",
            "varied_camera_1_right_image",
            "varied_camera_2_left_image",
            "varied_camera_2_right_image",
        ]

    def load_mp4_frames(self, mp4_path: str) -> np.ndarray:
        """
        Load all frames from an MP4 file.
        
        Args:
            mp4_path: Path to MP4 file
            
        Returns:
            Array of frames with shape (num_frames, height, width, channels)
        """
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

    def load_droid_trajectory(self, droid_path: str) -> Dict:
        """
        Load a DROID trajectory from downloaded files.

        Args:
            droid_path: Path to downloaded DROID trajectory directory

        Returns:
            Dictionary containing trajectory data
        """
        trajectory_data = {}

        # Load metadata
        metadata_path = None
        for file in os.listdir(droid_path):
            if file.startswith("metadata") and file.endswith(".json"):
                metadata_path = os.path.join(droid_path, file)
                break

        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                trajectory_data["metadata"] = json.load(f)

        # Load trajectory h5 file
        traj_path = os.path.join(droid_path, "trajectory.h5")
        if os.path.exists(traj_path):
            with h5py.File(traj_path, "r") as f:
                # Extract actions
                if "action" in f:
                    action_group = f["action"]
                    # Combine relevant action components
                    trajectory_data["actions"] = {
                        "joint_position":
                        np.array(action_group["joint_position"]),
                        "gripper_position":
                        np.array(action_group["gripper_position"]),
                        "cartesian_position":
                        np.array(action_group["cartesian_position"]),
                    }

                # Extract observations (proprioception)
                if "observation" in f:
                    obs_group = f["observation"]
                    trajectory_data["observations"] = {}
                    if "robot_state" in obs_group:
                        robot_state = obs_group["robot_state"]
                        for key in robot_state.keys():
                            trajectory_data["observations"][key] = np.array(
                                robot_state[key])

        # Load camera data from MP4 files
        trajectory_data["images"] = {}
        
        # Map MP4 files to camera names using metadata
        if "metadata" in trajectory_data:
            metadata = trajectory_data["metadata"]
            mp4_mappings = [
                ("wrist_mp4_path", "hand_camera_left_image"),
                ("ext1_mp4_path", "varied_camera_1_left_image"), 
                ("ext2_mp4_path", "varied_camera_2_left_image"),
            ]
            
            # Also handle stereo versions
            stereo_mappings = [
                ("wrist_mp4_path", "hand_camera_right_image"),
                ("ext1_mp4_path", "varied_camera_1_right_image"),
                ("ext2_mp4_path", "varied_camera_2_right_image"),
            ]
            
            for mp4_key, cam_name in mp4_mappings:
                if mp4_key in metadata:
                    mp4_path = os.path.join(droid_path, "recordings", "MP4", 
                                          os.path.basename(metadata[mp4_key]))
                    if os.path.exists(mp4_path):
                        images = self.load_mp4_frames(mp4_path)
                        if len(images) > 0:
                            trajectory_data["images"][cam_name] = images
                            print(f"  Loaded {cam_name}: shape {images.shape}")
                    
                    # Try stereo version
                    stereo_filename = os.path.basename(metadata[mp4_key]).replace(".mp4", "-stereo.mp4")
                    stereo_path = os.path.join(droid_path, "recordings", "MP4", stereo_filename)
                    if os.path.exists(stereo_path):
                        images = self.load_mp4_frames(stereo_path)
                        if len(images) > 0:
                            # For stereo, use right camera name
                            right_cam_name = cam_name.replace("left", "right")
                            trajectory_data["images"][right_cam_name] = images
                            print(f"  Loaded {right_cam_name}: shape {images.shape}")

        return trajectory_data

    def convert_to_robodm(self,
                          droid_data: Dict,
                          output_path: str,
                          video_codec: str = "libx264") -> Trajectory:
        """
        Convert DROID trajectory data to RoboDM format.

        Args:
            droid_data: Dictionary containing DROID trajectory data
            output_path: Path to save RoboDM trajectory
            video_codec: Video codec to use for compression

        Returns:
            RoboDM Trajectory object
        """
        # Create RoboDM trajectory
        traj = robodm.Trajectory(path=output_path, mode="w")

        # Determine trajectory length
        traj_len = 0
        if "actions" in droid_data and "joint_position" in droid_data[
                "actions"]:
            traj_len = len(droid_data["actions"]["joint_position"])
        elif "images" in droid_data:
            for cam_images in droid_data["images"].values():
                traj_len = len(cam_images)
                break

        print(f"  Converting {traj_len} timesteps to RoboDM format...")

        # Add data for each timestep
        for t in range(traj_len):
            # Add images from each camera
            for cam_name, images in droid_data["images"].items():
                if t < len(images):
                    traj.add(f"observation/images/{cam_name}", images[t])

            # Add actions
            if "actions" in droid_data:
                # Combine actions into single vector
                action_components = []
                if "joint_position" in droid_data["actions"] and t < len(
                        droid_data["actions"]["joint_position"]):
                    action_components.append(
                        droid_data["actions"]["joint_position"][t])
                if "gripper_position" in droid_data["actions"] and t < len(
                        droid_data["actions"]["gripper_position"]):
                    action_components.append(
                        [droid_data["actions"]["gripper_position"][t]])

                if action_components:
                    action = np.concatenate(action_components).astype(
                        np.float32)
                    traj.add("action", action)

            # Add proprioceptive observations
            if "observations" in droid_data:
                for obs_key, obs_data in droid_data["observations"].items():
                    if t < len(obs_data):
                        traj.add(
                            f"observation/state/{obs_key}",
                            obs_data[t].astype(np.float32),
                        )

        # Add metadata as regular data (RoboDM doesn't have set_metadata)
        if "metadata" in droid_data:
            # Store metadata as JSON string in a special key
            import json

            metadata_str = json.dumps(droid_data["metadata"])
            traj.add("metadata", metadata_str)

        traj.close()
        return traj

    def discover_trajectories(self, trajectory_type: str = "success", limit: int = None, labs: List[str] = None) -> List[str]:
        """
        Discover available trajectories from GCS using gsutil across all labs.
        
        Args:
            trajectory_type: Either "success" or "failure"
            limit: Maximum number of trajectories to return (None for all)
            labs: List of lab names to search (None for all available labs)
            
        Returns:
            List of trajectory paths
        """
        # Get all available labs if not specified
        if labs is None:
            try:
                result = subprocess.run(
                    ["gsutil", "ls", self.base_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                labs = [line.strip().rstrip('/').split('/')[-1] for line in result.stdout.strip().split('\n') 
                       if line.strip().endswith('/') and not line.strip().endswith('1.0.1/')]
                
            except subprocess.CalledProcessError as e:
                print(f"Error discovering labs: {e}")
                return []
        
        trajectories = []
        
        for lab in labs:
            lab_path = f"{self.base_path}{lab}/{trajectory_type}/"
            
            try:
                # Check if this lab has the trajectory type directory
                result = subprocess.run(
                    ["gsutil", "ls", lab_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                date_dirs = [line.strip() for line in result.stdout.strip().split('\n') 
                            if line.strip().endswith('/') and line.strip() != lab_path]
                
                # Get individual trajectories from each date directory
                for date_dir in date_dirs:
                    try:
                        date_result = subprocess.run(
                            ["gsutil", "ls", date_dir],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        date_trajectories = [line.strip() for line in date_result.stdout.strip().split('\n') 
                                           if line.strip().endswith('/')]
                        
                        trajectories.extend(date_trajectories)
                        
                        if limit and len(trajectories) >= limit:
                            break
                            
                    except subprocess.CalledProcessError:
                        continue
                        
                if limit and len(trajectories) >= limit:
                    break
                    
            except subprocess.CalledProcessError:
                # Lab doesn't have this trajectory type, skip
                continue
                
        return trajectories[:limit] if limit else trajectories

    def download_sample_trajectories(self,
                                     output_dir: str,
                                     num_success: int = 300,
                                     num_failure: int = 100):
        """
        Download and convert successful and failed trajectories in parallel from all labs.

        Args:
            output_dir: Directory to save RoboDM trajectories
            num_success: Number of successful trajectories to process
            num_failure: Number of failed trajectories to process
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="droid_download_")
        
        try:
            # Discover available trajectories from all labs
            print("Discovering available trajectories across all labs...")
            success_trajectories = self.discover_trajectories("success", limit=num_success * 2)  # Get more than needed
            failure_trajectories = self.discover_trajectories("failure", limit=num_failure * 2)  # Get more than needed
            
            print(f"Found {len(success_trajectories)} success trajectories")
            print(f"Found {len(failure_trajectories)} failure trajectories")

            # Curate the exact number requested
            selected_success = success_trajectories[:num_success]
            selected_failure = failure_trajectories[:num_failure]
            
            # Combine trajectories to process
            trajectories_to_process = selected_success + selected_failure

            print(f"Processing {len(trajectories_to_process)} trajectories in parallel...")
            print(f"  - {len(selected_success)} success trajectories")
            print(f"  - {len(selected_failure)} failure trajectories")
            
            # Submit all download and conversion tasks to Ray
            futures = []
            for traj_path in trajectories_to_process:
                future = download_and_convert_trajectory.remote(traj_path, output_dir, temp_dir)
                futures.append(future)

            # Process results as they complete
            completed = 0
            failed = 0
            successful_paths = []
            
            while futures:
                # Wait for at least one task to complete
                ready, futures = ray.wait(futures, num_returns=1)
                
                for future in ready:
                    success, output_path, error_msg = ray.get(future)
                    completed += 1
                    
                    if success:
                        successful_paths.append(output_path)
                        print(f"  [{completed}/{len(trajectories_to_process)}] Successfully processed to {output_path}")
                    else:
                        failed += 1
                        print(f"  [{completed}/{len(trajectories_to_process)}] Failed processing: {error_msg}")
            
            print(f"\nProcessing complete: {completed - failed} successful, {failed} failed")
            return successful_paths
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def convert_directory(self, input_dir: str, output_dir: str, max_workers: Optional[int] = None):
        """
        Convert all DROID trajectories in a directory to RoboDM format using Ray parallelization.
        This method is kept for backward compatibility when trajectories are already downloaded.

        Args:
            input_dir: Directory containing downloaded DROID trajectories
            output_dir: Directory to save RoboDM trajectories
            max_workers: Maximum number of parallel workers (None for automatic)
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        os.makedirs(output_dir, exist_ok=True)

        # Find all trajectory directories
        traj_dirs = []
        for root, dirs, files in os.walk(input_dir):
            if "trajectory.h5" in files:
                traj_dirs.append(root)

        print(f"Found {len(traj_dirs)} trajectories to convert")

        # Submit all conversion tasks to Ray
        print("Submitting conversion tasks to Ray...")
        futures = []
        for traj_dir in traj_dirs:
            future = convert_single_trajectory.remote(traj_dir, output_dir)
            futures.append(future)

        # Process results as they complete
        print("Processing trajectories in parallel...")
        completed = 0
        failed = 0
        
        while futures:
            # Wait for at least one task to complete
            ready, futures = ray.wait(futures, num_returns=1)
            
            for future in ready:
                success, output_path, error_msg = ray.get(future)
                completed += 1
                
                if success:
                    print(f"  [{completed}/{len(traj_dirs)}] Successfully converted to {output_path}")
                else:
                    failed += 1
                    print(f"  [{completed}/{len(traj_dirs)}] Failed conversion: {error_msg}")
        
        print(f"\nConversion complete: {completed - failed} successful, {failed} failed")
    
    def shutdown_ray(self):
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()


@ray.remote
def convert_single_trajectory(traj_dir: str, output_dir: str) -> Tuple[bool, str, str]:
    """
    Convert a single DROID trajectory to RoboDM format.
    This function is kept for backward compatibility when trajectories are already downloaded.
    
    Args:
        traj_dir: Path to DROID trajectory directory
        output_dir: Directory to save RoboDM trajectories
        
    Returns:
        Tuple of (success: bool, output_path: str, error_msg: str)
    """
    converter = DROIDProcessor()
    
    try:
        # Load DROID data
        droid_data = converter.load_droid_trajectory(traj_dir)
        
        # Generate output filename
        traj_name = os.path.basename(traj_dir)
        success_or_failure = "success" if "success" in traj_dir else "failure"
        output_path = os.path.join(output_dir, f"{success_or_failure}_{traj_name}.vla")
        
        # Convert to RoboDM
        converter.convert_to_robodm(droid_data, output_path)
        
        return True, output_path, ""
        
    except Exception as e:
        import traceback
        error_msg = f"Error converting {traj_dir}: {e}\n{traceback.format_exc()}"
        return False, "", error_msg


if __name__ == "__main__":
    # Example usage
    processor = DROIDProcessor()
    output_dir = "./robodm_trajectories"

    try:
        # Parallel download and conversion with 300 success + 100 failure trajectories
        print("Starting parallel download and conversion...")
        successful_paths = processor.download_sample_trajectories(
            output_dir=output_dir, 
            num_success=300, 
            num_failure=100
        )
        
        print(f"\nSuccessfully processed {len(successful_paths)} trajectories:")
        print(f"Output directory: {output_dir}")
        
        # Count success/failure trajectories
        success_count = len([p for p in successful_paths if "success_" in p])
        failure_count = len([p for p in successful_paths if "failure_" in p])
        print(f"  - {success_count} success trajectories")
        print(f"  - {failure_count} failure trajectories")
            
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Ensure Ray is properly shut down
        processor.shutdown_ray()
