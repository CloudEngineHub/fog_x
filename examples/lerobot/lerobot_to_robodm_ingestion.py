#!/usr/bin/env python3
"""
LeRobot to RoboDM Dataset Ingestion Pipeline

This module handles the conversion of LeRobot datasets to RoboDM format for parallel processing.
It provides a clean ingestion interface that can be used standalone or as part of a larger pipeline.

Usage:
    python lerobot_to_robodm_ingestion.py --dataset lerobot/pusht --num_episodes 50 --output_dir ./robodm_data
"""

import os
import tempfile
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# RoboDM imports
from robodm.trajectory import Trajectory

# LeRobot imports (if available)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    # Set backend to pyav for video processing
    import lerobot.datasets.video_utils as video_utils
    if hasattr(video_utils, 'set_video_backend'):
        video_utils.set_video_backend('pyav')
    LEROBOT_AVAILABLE = True
except ImportError:
    print("LeRobot not available. Please install lerobot package.")
    LEROBOT_AVAILABLE = False


class LeRobotToRoboDMIngestion:
    """Handles conversion of LeRobot datasets to RoboDM format."""
    
    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            dataset_name: Name of the LeRobot dataset (e.g., 'lerobot/pusht')
            output_dir: Directory to save RoboDM trajectories. If None, uses temp directory.
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is not available. Please install lerobot package.")
        
        self.dataset_name = dataset_name
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="robodm_lerobot_")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset metadata
        try:
            self.metadata = LeRobotDatasetMetadata(dataset_name)
            print(f"Dataset info: {self.metadata.total_episodes} episodes, {self.metadata.total_frames} frames")
        except Exception as e:
            print(f"Could not load metadata: {e}. Proceeding without metadata.")
            self.metadata = None
    
    def ingest(self, num_episodes: Optional[int] = None, video_backend: str = 'pyav') -> str:
        """
        Convert LeRobot dataset to RoboDM format.
        
        Args:
            num_episodes: Number of episodes to convert. If None, converts all episodes.
            video_backend: Video backend to use for processing ('pyav' or 'opencv').
        
        Returns:
            Path to the directory containing converted RoboDM trajectories.
        """
        print(f"Starting ingestion of {self.dataset_name}")
        print(f"Output directory: {self.output_dir}")
        
        # Determine episodes to load
        episodes_to_load = None
        if num_episodes is not None and self.metadata is not None:
            episodes_to_load = list(range(min(num_episodes, self.metadata.total_episodes)))
        
        # Load LeRobot dataset
        print(f"Loading dataset with episodes: {episodes_to_load if episodes_to_load else 'all'}")
        lerobot_dataset = self._load_lerobot_dataset(episodes_to_load, video_backend)
        
        # Convert to RoboDM format
        self._convert_to_robodm(lerobot_dataset)
        
        print(f"✅ Ingestion completed successfully!")
        print(f"RoboDM trajectories saved to: {self.output_dir}")
        return self.output_dir
    
    def _load_lerobot_dataset(self, episodes_to_load: Optional[list], video_backend: str) -> LeRobotDataset:
        """Load LeRobot dataset with proper video backend."""
        try:
            dataset = LeRobotDataset(
                self.dataset_name, 
                episodes=episodes_to_load, 
                video_backend=video_backend
            )
        except TypeError:
            # Fallback if video_backend parameter is not supported
            dataset = LeRobotDataset(self.dataset_name, episodes=episodes_to_load)
        
        print(f"Dataset loaded with {len(dataset)} samples")
        return dataset
    
    def _convert_to_robodm(self, lerobot_dataset: LeRobotDataset):
        """Convert LeRobot dataset to RoboDM trajectory format."""
        # Group samples by episode
        episodes_data = {}
        for i, sample in enumerate(lerobot_dataset):
            episode_idx = sample['episode_index'].item()
            frame_idx = sample['frame_index'].item()
            
            if episode_idx not in episodes_data:
                episodes_data[episode_idx] = []
            
            episodes_data[episode_idx].append((frame_idx, sample))
        
        # Sort each episode by frame index
        for episode_idx in episodes_data:
            episodes_data[episode_idx].sort(key=lambda x: x[0])
        
        print(f"Converting {len(episodes_data)} episodes to RoboDM format...")
        
        # Convert each episode to RoboDM trajectory
        for episode_idx, frames in episodes_data.items():
            self._convert_episode_to_trajectory(episode_idx, frames)
    
    def _convert_episode_to_trajectory(self, episode_idx: int, frames: list):
        """Convert a single episode to a RoboDM trajectory file."""
        trajectory_path = os.path.join(self.output_dir, f"episode_{episode_idx:03d}.vla")
        traj = Trajectory(path=trajectory_path, mode="w")
        
        try:
            for frame_idx, sample in frames:
                # Convert timestamp (assuming 10 FPS by default)
                timestamp = frame_idx * 100  # 100ms intervals = 10 FPS
                
                # Add image observations
                self._add_image_observations(traj, sample, timestamp)
                
                # Add state observations
                if 'observation.state' in sample:
                    state = sample['observation.state'].numpy().astype(np.float32)
                    traj.add("observation/state", state, timestamp=timestamp, time_unit="ms")
                
                # Add actions
                if 'action' in sample:
                    action = sample['action'].numpy().astype(np.float32)
                    traj.add("action", action, timestamp=timestamp, time_unit="ms")
                
                # Add reward and done signals if available
                if 'next.reward' in sample:
                    reward = sample['next.reward'].numpy().astype(np.float32)
                    traj.add("reward", reward, timestamp=timestamp, time_unit="ms")
                
                if 'next.done' in sample:
                    done = sample['next.done'].numpy().astype(np.bool_)
                    traj.add("done", done, timestamp=timestamp, time_unit="ms")
        
        finally:
            traj.close()
    
    def _add_image_observations(self, traj: Trajectory, sample: Dict[str, Any], timestamp: int):
        """Add image observations to trajectory."""
        # Handle primary image observation
        if 'observation.image' in sample:
            image = sample['observation.image'].permute(1, 2, 0).numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            traj.add("observation/image", image, timestamp=timestamp, time_unit="ms")
        
        # Handle multiple camera observations
        for key in sample.keys():
            if key.startswith('observation.images.'):
                camera_name = key.split('.')[-1]
                image = sample[key].permute(1, 2, 0).numpy()
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                traj.add(f"observation/images/{camera_name}", image, timestamp=timestamp, time_unit="ms")
            elif key.startswith('observation.image') and key != 'observation.image':
                # Handle other image observations like observation.image_front, etc.
                camera_name = key.split('.')[-1] if '.' in key else key.replace('observation.', '')
                image = sample[key].permute(1, 2, 0).numpy()
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                traj.add(f"observation/images/{camera_name}", image, timestamp=timestamp, time_unit="ms")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about the converted dataset."""
        trajectory_files = list(Path(self.output_dir).glob("*.vla"))
        return {
            "output_directory": self.output_dir,
            "num_trajectories": len(trajectory_files),
            "trajectory_files": [str(f) for f in trajectory_files],
            "total_size_mb": sum(f.stat().st_size for f in trajectory_files) / (1024 * 1024)
        }


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to RoboDM format")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="LeRobot dataset name (e.g., lerobot/pusht)")
    parser.add_argument("--num_episodes", type=int, default=None, 
                       help="Number of episodes to convert (None for all)")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory for RoboDM trajectories")
    parser.add_argument("--video_backend", type=str, default='pyav', 
                       choices=['pyav', 'opencv'], help="Video backend to use")
    
    args = parser.parse_args()
    
    # Create ingestion pipeline
    ingestion = LeRobotToRoboDMIngestion(
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    # Run ingestion
    output_dir = ingestion.ingest(
        num_episodes=args.num_episodes,
        video_backend=args.video_backend
    )
    
    # Print statistics
    stats = ingestion.get_conversion_stats()
    print(f"\n📊 Conversion Statistics:")
    print(f"  Output directory: {stats['output_directory']}")
    print(f"  Trajectories converted: {stats['num_trajectories']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()