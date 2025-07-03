import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

import robodm
from robodm import Trajectory


class DROIDToRoboDMConverter:
    """Converts DROID trajectories to RoboDM format."""

    def __init__(self):
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

    def convert_directory(self, input_dir: str, output_dir: str):
        """
        Convert all DROID trajectories in a directory to RoboDM format.

        Args:
            input_dir: Directory containing downloaded DROID trajectories
            output_dir: Directory to save RoboDM trajectories
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all trajectory directories
        traj_dirs = []
        for root, dirs, files in os.walk(input_dir):
            if "trajectory.h5" in files:
                traj_dirs.append(root)

        print(f"Found {len(traj_dirs)} trajectories to convert")

        # Convert each trajectory
        for i, traj_dir in enumerate(traj_dirs):
            print(
                f"\nConverting trajectory {i+1}/{len(traj_dirs)}: {traj_dir}")

            try:
                # Load DROID data
                droid_data = self.load_droid_trajectory(traj_dir)

                # Generate output filename
                traj_name = os.path.basename(traj_dir)
                success_or_failure = "success" if "success" in traj_dir else "failure"
                output_path = os.path.join(
                    output_dir, f"{success_or_failure}_{traj_name}.vla")

                # Convert to RoboDM
                self.convert_to_robodm(droid_data, output_path)
                print(f"  Successfully converted to {output_path}")

            except Exception as e:
                print(f"  Error converting {traj_dir}: {e}")
                import traceback

                traceback.print_exc()
                continue


if __name__ == "__main__":
    # Example usage
    converter = DROIDToRoboDMConverter()

    # Convert downloaded DROID trajectories
    input_dir = "./droid_data"
    output_dir = "./robodm_trajectories"

    if os.path.exists(input_dir):
        converter.convert_directory(input_dir, output_dir)
    else:
        print(
            f"Input directory {input_dir} not found. Please run download_droid.py first."
        )
