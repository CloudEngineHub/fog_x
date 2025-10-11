#!/usr/bin/env python3
"""
DROID Trajectory Download Example

Concise example for downloading DROID trajectories to local storage with parallel processing.
Downloads trajectories from GCS and converts them to robodm format for efficient processing.

Usage:
    python droid_download_example.py --gcs-pattern "gs://gresearch/robotics/droid_raw/1.0.1/*/success/*" --local-dir ./droid_data --num-trajectories 50
"""

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import ray

logger = logging.getLogger(__name__)


@ray.remote
def download_single_trajectory(gcs_path: str, local_dir: str, temp_dir: str) -> Tuple[bool, Optional[str], str, str]:
    """Download a single DROID trajectory from GCS to local directory."""
    try:
        # Extract meaningful name from nested structure: date/trajectory_name
        path_parts = gcs_path.rstrip("/").split("/")
        date_part = path_parts[-2]  # e.g., "2023-07-07"
        traj_part = path_parts[-1]  # e.g., "Fri_Jul__7_09:42:23_2023"
        trajectory_name = f"{date_part}_{traj_part}"
        local_trajectory_dir = Path(local_dir) / trajectory_name
        local_trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # Use gsutil for efficient GCS download
        # Remove trailing slash and add /* for contents
        clean_gcs_path = gcs_path.rstrip("/")
        cmd = ["gsutil", "-m", "cp", "-r", f"{clean_gcs_path}/*", str(local_trajectory_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Downloaded {trajectory_name}")
            return True, str(local_trajectory_dir), "", trajectory_name
        else:
            error_msg = f"gsutil failed: {result.stderr}"
            logger.error(f"Failed to download {trajectory_name}: {error_msg}")
            return False, None, error_msg, trajectory_name
            
    except Exception as e:
        error_msg = f"Exception during download: {str(e)}"
        logger.error(f"Error downloading {gcs_path}: {error_msg}")
        return False, None, error_msg, trajectory_name


def scan_droid_trajectories(gcs_pattern: str, max_trajectories: Optional[int] = None) -> List[str]:
    """Scan for DROID trajectories matching the GCS pattern."""
    try:
        # First get date directories
        cmd = ["gsutil", "ls", "-d", gcs_pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"gsutil ls failed: {result.stderr}")
            
        date_dirs = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        # Now get actual trajectory directories from each date
        all_trajectories = []
        for date_dir in date_dirs:
            if max_trajectories and len(all_trajectories) >= max_trajectories:
                break
                
            cmd = ["gsutil", "ls", "-d", f"{date_dir.rstrip('/')}/*"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                traj_dirs = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                all_trajectories.extend(traj_dirs)
        
        if max_trajectories and len(all_trajectories) > max_trajectories:
            all_trajectories = all_trajectories[:max_trajectories]
            
        return all_trajectories
        
    except Exception as e:
        logger.error(f"Failed to scan trajectories: {e}")
        return []


def download_droid_trajectories(
    gcs_pattern: str,
    local_dir: str,
    num_trajectories: Optional[int] = None,
    parallel_downloads: int = 8
) -> Tuple[List[str], List[str]]:
    """
    Download DROID trajectories from GCS to local directory.
    
    Args:
        gcs_pattern: GCS pattern for trajectory paths (e.g., "gs://path/*/success/*")
        local_dir: Local directory to store downloaded trajectories
        num_trajectories: Maximum number of trajectories to download
        parallel_downloads: Number of parallel downloads
        
    Returns:
        Tuple of (successful_paths, failed_paths)
    """
    if not ray.is_initialized():
        ray.init()
    
    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Scan for trajectories
    print(f"Scanning for trajectories matching: {gcs_pattern}")
    trajectory_paths = scan_droid_trajectories(gcs_pattern, num_trajectories)
    
    if not trajectory_paths:
        print("No trajectories found matching the pattern")
        return [], []
        
    print(f"Found {len(trajectory_paths)} trajectories to download")
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # Start parallel downloads
        print(f"Starting {parallel_downloads} parallel downloads...")
        
        download_futures = []
        for gcs_path in trajectory_paths:
            future = download_single_trajectory.remote(gcs_path, local_dir, temp_dir)
            download_futures.append((future, gcs_path))
            
        # Process results as they complete
        successful_paths = []
        failed_paths = []
        
        for future, gcs_path in download_futures:
            try:
                success, local_path, error_msg, traj_name = ray.get(future)
                if success:
                    successful_paths.append(local_path)
                    print(f"✅ {traj_name}")
                else:
                    failed_paths.append(gcs_path)
                    print(f"❌ {traj_name}: {error_msg}")
            except Exception as e:
                failed_paths.append(gcs_path)
                print(f"❌ {gcs_path}: {e}")
    
    print(f"\nDownload complete: {len(successful_paths)} successful, {len(failed_paths)} failed")
    return successful_paths, failed_paths


def main():
    """Download DROID trajectories from GCS."""
    parser = argparse.ArgumentParser(description="Download DROID trajectories from GCS")
    parser.add_argument("--gcs-pattern", default = "gs://gresearch/robotics/droid_raw/1.0.1/*/success/*",
                       help="GCS pattern for trajectory paths")
    parser.add_argument("--local-dir", required=True,
                       help="Local directory to store trajectories")
    parser.add_argument("--num-trajectories", type=int, default=None,
                       help="Maximum number of trajectories to download")
    parser.add_argument("--parallel-downloads", type=int, default=8,
                       help="Number of parallel downloads")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("DROID Trajectory Downloader")
    print("=" * 50)
    print(f"GCS pattern: {args.gcs_pattern}")
    print(f"Local directory: {args.local_dir}")
    print(f"Max trajectories: {args.num_trajectories or 'All'}")
    print(f"Parallel downloads: {args.parallel_downloads}")
    print()
    
    # Download trajectories
    successful_paths, failed_paths = download_droid_trajectories(
        gcs_pattern=args.gcs_pattern,
        local_dir=args.local_dir,
        num_trajectories=args.num_trajectories,
        parallel_downloads=args.parallel_downloads
    )
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"Successful: {len(successful_paths)}")
    print(f"Failed: {len(failed_paths)}")
    
    if successful_paths:
        print(f"\nTrajectories saved to: {args.local_dir}")
        print("Ready for processing with robodm!")


if __name__ == "__main__":
    main()