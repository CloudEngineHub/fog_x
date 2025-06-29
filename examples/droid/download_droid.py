import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import h5py


class DROIDDownloader:
    """Downloads DROID trajectories from Google Cloud Storage."""

    def __init__(self,
                 base_path: str = "gs://gresearch/robotics/droid_raw/1.0.1/"):
        self.base_path = base_path

    def download_trajectory(self, trajectory_path: str,
                            output_dir: str) -> str:
        """
        Download a single trajectory from GCS.

        Args:
            trajectory_path: Full GCS path to trajectory
            output_dir: Local directory to save trajectory

        Returns:
            Path to downloaded trajectory directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract trajectory name from path
        traj_name = trajectory_path.rstrip("/").split("/")[-1]
        local_path = os.path.join(output_dir, traj_name)

        # Download using gsutil
        print(f"Downloading {trajectory_path} to {local_path}")
        try:
            # gsutil needs the parent directory to exist
            parent_dir = os.path.dirname(local_path)
            os.makedirs(parent_dir, exist_ok=True)

            subprocess.run(
                ["gsutil", "-m", "cp", "-r", trajectory_path, parent_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Successfully downloaded to {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            print(f"Error downloading trajectory: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None

    def download_sample_trajectories(self,
                                     output_dir: str,
                                     num_success: int = 2,
                                     num_failure: int = 2):
        """
        Download sample successful and failed trajectories.

        Args:
            output_dir: Directory to save trajectories
            num_success: Number of successful trajectories to download
            num_failure: Number of failed trajectories to download
        """
        # Sample trajectory paths - using ones we verified exist
        success_trajectories = [
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-07-07/Fri_Jul__7_09:43:39_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-07-08/Sat_Jul__8_08:57:28_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-07-08/Sat_Jul__8_08:59:35_2023/",
        ]

        failure_trajectories = [
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_09:45:39_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_09:48:37_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_09:49:13_2023/",
            "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_09:50:13_2023/",
        ]

        # Create success and failure directories
        success_dir = os.path.join(output_dir, "success")
        failure_dir = os.path.join(output_dir, "failure")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failure_dir, exist_ok=True)

        # Download successful trajectories
        print(f"\nDownloading {num_success} successful trajectories...")
        downloaded_success = []
        for i, traj_path in enumerate(success_trajectories[:num_success]):
            local_path = self.download_trajectory(traj_path, success_dir)
            if local_path:
                downloaded_success.append(local_path)

        # Download failed trajectories
        print(f"\nDownloading {num_failure} failed trajectories...")
        downloaded_failure = []
        for i, traj_path in enumerate(failure_trajectories[:num_failure]):
            local_path = self.download_trajectory(traj_path, failure_dir)
            if local_path:
                downloaded_failure.append(local_path)

        return downloaded_success, downloaded_failure


if __name__ == "__main__":
    # Example usage
    downloader = DROIDDownloader()

    # Download sample trajectories
    output_dir = "./droid_data"
    success_paths, failure_paths = downloader.download_sample_trajectories(
        output_dir=output_dir, num_success=2, num_failure=2)

    print(f"\nDownloaded {len(success_paths)} successful trajectories")
    print(f"Downloaded {len(failure_paths)} failed trajectories")
