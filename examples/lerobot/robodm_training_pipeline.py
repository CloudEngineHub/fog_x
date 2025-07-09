#!/usr/bin/env python3
"""
RoboDM Training Pipeline

This module provides a bridge between RoboDM datasets and LeRobot training pipeline.
It handles loading RoboDM datasets, converting them to LeRobot format, and providing
the necessary interfaces for training.

Usage:
    from robodm_training_pipeline import RoboDMTrainingPipeline
    
    pipeline = RoboDMTrainingPipeline(robodm_data_dir="./robodm_data")
    torch_dataset = pipeline.get_torch_dataset()
    # Use torch_dataset with LeRobot training
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.utils.data as torch_data

# RoboDM imports
from robodm.dataset import VLADataset, DatasetConfig

# LeRobot imports (if available)
try:
    from lerobot.configs.types import FeatureType, PolicyFeature
    LEROBOT_AVAILABLE = True
except ImportError:
    print("LeRobot not available. Some features will be limited.")
    LEROBOT_AVAILABLE = False


class RoboDMToLeRobotBridge:
    """Bridge to convert RoboDM data to LeRobot format for training."""
    
    def __init__(self, robodm_dataset: VLADataset):
        """
        Initialize the bridge with a RoboDM dataset.
        
        Args:
            robodm_dataset: Loaded RoboDM VLADataset instance
        """
        self.robodm_dataset = robodm_dataset
        
        # Load trajectories if not already loaded
        if not robodm_dataset._is_loaded:
            print("Loading trajectories for bridge...")
            self.robodm_dataset = robodm_dataset.load_trajectories()
        
        # Create PyTorch dataset
        print("Creating PyTorch dataset from RoboDM data...")
        self.torch_dataset = self._create_torch_dataset()
    
    def _create_torch_dataset(self) -> torch_data.Dataset:
        """Convert RoboDM dataset to PyTorch dataset."""
        # Get all trajectories - properly materialize the dataset
        ray_dataset = self.robodm_dataset.get_ray_dataset()
        trajectories = list(ray_dataset.iter_rows())
        
        print(f"Converting {len(trajectories)} trajectories...")
        
        # Convert each trajectory to timesteps
        all_timesteps = []
        for episode_idx, traj in enumerate(trajectories):
            try:
                timesteps = self._convert_trajectory(traj, episode_idx)
                all_timesteps.extend(timesteps)
                if (episode_idx + 1) % 10 == 0:
                    print(f"  Processed {episode_idx + 1}/{len(trajectories)} trajectories")
            except Exception as e:
                print(f"  Warning: Failed to convert trajectory {episode_idx}: {e}")
                continue
        
        print(f"Created dataset with {len(all_timesteps)} timesteps")
        return SimplePyTorchDataset(all_timesteps)
    
    def _convert_trajectory(self, trajectory: Dict[str, Any], episode_idx: int) -> List[Dict[str, torch.Tensor]]:
        """Convert single trajectory to list of timesteps with action sequences."""
        # Find trajectory length from available data
        traj_len = 0
        image_keys = [k for k in trajectory.keys() if 'observation/image' in k or 'observation/images' in k]
        state_keys = [k for k in trajectory.keys() if 'observation/state' in k]
        action_keys = [k for k in trajectory.keys() if 'action' in k]
        
        # Determine trajectory length from the first available data source
        if image_keys and len(trajectory[image_keys[0]]) > 0:
            traj_len = len(trajectory[image_keys[0]])
        elif action_keys and len(trajectory[action_keys[0]]) > 0:
            traj_len = len(trajectory[action_keys[0]])
        elif state_keys and len(trajectory[state_keys[0]]) > 0:
            traj_len = len(trajectory[state_keys[0]])
        else:
            return []  # No valid data found
        
        # DiffusionPolicy expects sequences with full prediction horizon
        horizon = 16  # This should match DiffusionPolicy's horizon (not n_action_steps)
        timesteps = []
        
        # Create training samples with action sequences
        for frame_idx in range(traj_len - horizon + 1):  # Ensure we have enough future actions
            # Create timestep data in LeRobot format
            timestep = {
                'timestamp': torch.tensor([frame_idx * 0.1], dtype=torch.float32),  # 10 FPS
                'frame_index': torch.tensor([frame_idx], dtype=torch.int64),
                'episode_index': torch.tensor([episode_idx], dtype=torch.int64),
                'index': torch.tensor([len(timesteps)], dtype=torch.int64),
                'task_index': torch.tensor([0], dtype=torch.int64),
            }
            
            # Add single image observations (sequencing handled during batching)
            self._add_image_observations(timestep, trajectory, image_keys, frame_idx)
            
            # Add state observations
            if state_keys:
                state_data = trajectory[state_keys[0]][frame_idx] if frame_idx < len(trajectory[state_keys[0]]) else np.array([])
                if isinstance(state_data, np.ndarray) and len(state_data) > 0:
                    state_data = state_data.copy()  # Make writable
                    timestep['observation.state'] = torch.from_numpy(state_data).float()
                else:
                    # Create a placeholder if no state data
                    timestep['observation.state'] = torch.zeros(1, dtype=torch.float32)
            
            # Add action sequences (horizon length)
            if action_keys:
                action_sequence = []
                action_is_pad_sequence = []
                
                for action_idx in range(horizon):
                    seq_frame_idx = frame_idx + action_idx
                    if seq_frame_idx < len(trajectory[action_keys[0]]):
                        action_data = trajectory[action_keys[0]][seq_frame_idx]
                        if isinstance(action_data, np.ndarray) and len(action_data) > 0:
                            action_data = action_data.copy()  # Make writable
                            action_sequence.append(torch.from_numpy(action_data).float())
                            action_is_pad_sequence.append(False)
                        else:
                            # Pad with zeros
                            action_dim = action_data.shape[0] if hasattr(action_data, 'shape') else 2
                            action_sequence.append(torch.zeros(action_dim, dtype=torch.float32))
                            action_is_pad_sequence.append(True)
                    else:
                        # Pad with zeros when we run out of actions
                        action_dim = action_sequence[0].shape[0] if action_sequence else 2
                        action_sequence.append(torch.zeros(action_dim, dtype=torch.float32))
                        action_is_pad_sequence.append(True)
                
                # Stack into sequence tensors
                timestep['action'] = torch.stack(action_sequence)  # Shape: [horizon, action_dim]
                timestep['action_is_pad'] = torch.tensor(action_is_pad_sequence, dtype=torch.bool)  # Shape: [horizon]
            else:
                # No action data at all - use default action dimension
                default_action_dim = 2  # You should adjust this to match your robot's action space
                timestep['action'] = torch.zeros(horizon, default_action_dim, dtype=torch.float32)  # Shape: [horizon, action_dim]
                timestep['action_is_pad'] = torch.ones(horizon, dtype=torch.bool)  # All padded
            
            timesteps.append(timestep)
        
        return timesteps
    
    def _add_image_observation_sequences(self, timestep: Dict[str, torch.Tensor], trajectory: Dict[str, Any], 
                                        image_keys: List[str], frame_idx: int):
        """Add image observation sequences to timestep for DiffusionPolicy (n_obs_steps=2)."""
        n_obs_steps = 2  # DiffusionPolicy default
        
        if image_keys:
            primary_image_key = image_keys[0]  # Use first available image
            image_sequence = []
            
            # Collect n_obs_steps frames (current and previous)
            for obs_idx in range(n_obs_steps):
                obs_frame_idx = frame_idx - (n_obs_steps - 1 - obs_idx)  # Go backwards in time
                
                if obs_frame_idx >= 0 and obs_frame_idx < len(trajectory[primary_image_key]):
                    image_data = trajectory[primary_image_key][obs_frame_idx]
                    if isinstance(image_data, np.ndarray) and image_data.size > 0:
                        # Make a copy to ensure the array is writable
                        image_data = image_data.copy()
                        # Convert to tensor, ensure it's in CHW format
                        if len(image_data.shape) == 3:
                            # Check if it's HWC format (height, width, channels)
                            if image_data.shape[2] == 3:  # HWC format
                                image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0
                            elif image_data.shape[0] == 3:  # Already CHW format
                                image_tensor = torch.from_numpy(image_data).float() / 255.0
                            else:
                                # Unknown format, assume HWC and convert
                                image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0
                        else:
                            # Handle 2D images by adding channel dimension
                            if len(image_data.shape) == 2:
                                image_tensor = torch.from_numpy(image_data).unsqueeze(0).float() / 255.0
                            else:
                                # Fallback: try to reshape to CHW format
                                image_tensor = torch.from_numpy(image_data).float() / 255.0
                                if image_tensor.dim() == 1:
                                    # Try to reshape to square image
                                    size = int(np.sqrt(image_tensor.shape[0] / 3))
                                    if size * size * 3 == image_tensor.shape[0]:
                                        image_tensor = image_tensor.view(3, size, size)
                                    else:
                                        # Create placeholder if can't reshape
                                        image_tensor = torch.zeros(3, 96, 96, dtype=torch.float32)
                        image_sequence.append(image_tensor)
                    else:
                        # Create a placeholder image if no image data
                        image_sequence.append(torch.zeros(3, 96, 96, dtype=torch.float32))
                else:
                    # Create a placeholder image if frame is out of range (repeat first available frame)
                    if obs_frame_idx < 0 and len(image_sequence) > 0:
                        image_sequence.append(image_sequence[0].clone())  # Repeat first frame
                    else:
                        image_sequence.append(torch.zeros(3, 96, 96, dtype=torch.float32))
            
            # Stack into sequence format: (n_obs_steps, num_cameras, C, H, W)
            # For now, assume single camera, so shape will be: (n_obs_steps, 1, C, H, W)
            image_stack = torch.stack(image_sequence, dim=0)  # Shape: [n_obs_steps, C, H, W]
            image_stack = image_stack.unsqueeze(1)  # Add camera dimension: [n_obs_steps, 1, C, H, W]
            timestep['observation.images'] = image_stack  # Use 'images' not 'image'
        else:
            # No image data available
            timestep['observation.images'] = torch.zeros(n_obs_steps, 1, 3, 96, 96, dtype=torch.float32)
    
    def _add_image_observations(self, timestep: Dict[str, torch.Tensor], trajectory: Dict[str, Any], 
                               image_keys: List[str], frame_idx: int):
        """Add single image observations to timestep (legacy method)."""
        if image_keys:
            primary_image_key = image_keys[0]  # Use first available image
            if frame_idx < len(trajectory[primary_image_key]):
                image_data = trajectory[primary_image_key][frame_idx]
                if isinstance(image_data, np.ndarray) and image_data.size > 0:
                    # Make a copy to ensure the array is writable
                    image_data = image_data.copy()
                    # Convert to tensor, ensure it's in CHW format
                    if len(image_data.shape) == 3:
                        # Check if it's HWC format (height, width, channels)
                        if image_data.shape[2] == 3:  # HWC format
                            image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0
                        elif image_data.shape[0] == 3:  # Already CHW format
                            image_tensor = torch.from_numpy(image_data).float() / 255.0
                        else:
                            # Unknown format, assume HWC and convert
                            image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0
                    else:
                        # Handle 2D images by adding channel dimension
                        if len(image_data.shape) == 2:
                            image_tensor = torch.from_numpy(image_data).unsqueeze(0).float() / 255.0
                        else:
                            # Fallback: try to reshape to CHW format
                            image_tensor = torch.from_numpy(image_data).float() / 255.0
                            if image_tensor.dim() == 1:
                                # Try to reshape to square image
                                size = int(np.sqrt(image_tensor.shape[0] / 3))
                                if size * size * 3 == image_tensor.shape[0]:
                                    image_tensor = image_tensor.view(3, size, size)
                                else:
                                    # Create placeholder if can't reshape
                                    image_tensor = torch.zeros(3, 96, 96, dtype=torch.float32)
                    timestep['observation.image'] = image_tensor
                else:
                    # Create a placeholder image if no image data
                    timestep['observation.image'] = torch.zeros(3, 96, 96, dtype=torch.float32)
            else:
                # Create a placeholder image if frame is out of range
                timestep['observation.image'] = torch.zeros(3, 96, 96, dtype=torch.float32)
    
    def get_torch_dataset(self) -> torch_data.Dataset:
        """Get PyTorch dataset."""
        return self.torch_dataset
    
    def get_features_info(self) -> Dict[str, Dict[str, Any]]:
        """Get feature information for LeRobot policy configuration."""
        if len(self.torch_dataset) == 0:
            raise ValueError("Dataset is empty, cannot extract features")
        
        sample = self.torch_dataset[0]
        features = {}
        
        for key, value in sample.items():
            if key in ['observation.image', 'observation.state', 'action'] and value is not None:
                features[key] = {
                    'dtype': 'image' if 'image' in key else 'float32',
                    'shape': list(value.shape),
                    'names': None
                }
        
        return features
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Calculate dataset statistics for normalization."""
        print("Calculating dataset statistics...")
        
        # Collect all data
        all_images = []
        all_states = []
        all_actions = []
        
        for i, item in enumerate(self.torch_dataset):
            try:
                if 'observation.image' in item and item['observation.image'] is not None and hasattr(item['observation.image'], 'shape'):
                    all_images.append(item['observation.image'])
                if 'observation.state' in item and item['observation.state'] is not None and hasattr(item['observation.state'], 'shape'):
                    all_states.append(item['observation.state'])
                if 'action' in item and item['action'] is not None and hasattr(item['action'], 'shape'):
                    all_actions.append(item['action'])
            except Exception as e:
                print(f"Warning: Failed to process item {i}: {e}")
                continue
        
        stats = {}
        
        # Calculate stats for each data type
        if all_images:
            try:
                images = torch.stack(all_images)
                stats['observation.image'] = {
                    'mean': images.mean(dim=0),
                    'std': images.std(dim=0),
                    'min': images.min(dim=0)[0],
                    'max': images.max(dim=0)[0]
                }
            except Exception as e:
                print(f"Warning: Failed to calculate image stats: {e}")
        
        if all_states:
            try:
                states = torch.stack(all_states)
                stats['observation.state'] = {
                    'mean': states.mean(dim=0),
                    'std': states.std(dim=0),
                    'min': states.min(dim=0)[0],
                    'max': states.max(dim=0)[0]
                }
            except Exception as e:
                print(f"Warning: Failed to calculate state stats: {e}")
        
        if all_actions:
            try:
                actions = torch.stack(all_actions)
                # Transpose actions from [samples, horizon, action_dim] to [samples, action_dim, horizon]
                # to match the expected format for DiffusionPolicy
                if len(actions.shape) == 3:
                    actions = actions.transpose(1, 2)  # [samples, action_dim, horizon]
                stats['action'] = {
                    'mean': actions.mean(dim=0),
                    'std': actions.std(dim=0),
                    'min': actions.min(dim=0)[0],
                    'max': actions.max(dim=0)[0]
                }
            except Exception as e:
                print(f"Warning: Failed to calculate action stats: {e}")
        
        if not stats:
            print("Warning: No valid statistics calculated, using default stats")
            # Provide default stats if none calculated
            stats = {
                'observation.image': {
                    'mean': torch.zeros(3, 96, 96),
                    'std': torch.ones(3, 96, 96),
                    'min': torch.zeros(3, 96, 96),
                    'max': torch.ones(3, 96, 96)
                },
                'observation.state': {
                    'mean': torch.zeros(1),
                    'std': torch.ones(1),
                    'min': torch.zeros(1),
                    'max': torch.ones(1)
                },
                'action': {
                    'mean': torch.zeros(1),
                    'std': torch.ones(1),
                    'min': torch.zeros(1),
                    'max': torch.ones(1)
                }
            }
        
        return stats
    
    def get_policy_features(self) -> Dict[str, Dict[str, Any]]:
        """Get input and output features for policy configuration."""
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is required for policy feature extraction")
        
        features_info = self.get_features_info()
        
        if not features_info:
            raise ValueError("No valid features found in dataset")
        
        input_features = {}
        output_features = {}
        
        for key, info in features_info.items():
            feature = PolicyFeature(
                type=FeatureType.VISUAL if info['dtype'] == 'image' else FeatureType.STATE,
                shape=info['shape']
            )
            
            if 'action' in key:
                # Actions should use ACTION type, not STATE type
                feature = PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=info['shape']
                )
                output_features[key] = feature
            else:
                input_features[key] = feature
        
        if not output_features:
            raise ValueError("No action features found in dataset")
        if not input_features:
            raise ValueError("No observation features found in dataset")
        
        return {
            'input_features': input_features,
            'output_features': output_features
        }


class SimplePyTorchDataset(torch_data.Dataset):
    """Simple PyTorch dataset wrapper."""
    
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class RoboDMTrainingPipeline:
    """Complete training pipeline for RoboDM datasets."""
    
    def __init__(self, robodm_data_dir: str, config: Optional[DatasetConfig] = None):
        """
        Initialize the training pipeline.
        
        Args:
            robodm_data_dir: Directory containing RoboDM .vla files
            config: Optional DatasetConfig for customizing dataset loading
        """
        self.robodm_data_dir = robodm_data_dir
        self.config = config or DatasetConfig(
            batch_size=4,
            shuffle=False,
            num_parallel_reads=2,
            use_metadata=False,
        )
        
        # Load RoboDM dataset
        self.robodm_dataset = self._load_robodm_dataset()
        
        # Create bridge to LeRobot format
        self.bridge = RoboDMToLeRobotBridge(self.robodm_dataset)
    
    def _load_robodm_dataset(self) -> VLADataset:
        """Load RoboDM dataset from directory."""
        print(f"Loading RoboDM dataset from: {self.robodm_data_dir}")
        
        dataset = VLADataset(
            path=f"{self.robodm_data_dir}/*.vla",
            return_type="numpy",
            config=self.config
        )
        
        print(f"Found {dataset.count()} trajectory files")
        
        # Load trajectories in parallel
        print("Loading trajectories in parallel...")
        loaded_dataset = dataset.load_trajectories()
        
        print(f"✅ Loaded dataset with {loaded_dataset.count()} trajectories")
        return loaded_dataset
    
    def get_torch_dataset(self) -> torch_data.Dataset:
        """Get PyTorch dataset ready for training."""
        return self.bridge.get_torch_dataset()
    
    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True, 
                      num_workers: int = 4, **kwargs) -> torch_data.DataLoader:
        """Get PyTorch DataLoader for training."""
        return torch_data.DataLoader(
            self.get_torch_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            **kwargs
        )
    
    def get_features_info(self) -> Dict[str, Dict[str, Any]]:
        """Get feature information for policy configuration."""
        return self.bridge.get_features_info()
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get dataset statistics for normalization."""
        return self.bridge.get_dataset_stats()
    
    def get_policy_features(self) -> Dict[str, Dict[str, Any]]:
        """Get input and output features for policy configuration."""
        return self.bridge.get_policy_features()
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        torch_dataset = self.get_torch_dataset()
        features_info = self.get_features_info()
        
        return {
            'dataset_size': len(torch_dataset),
            'num_trajectories': self.robodm_dataset.count(),
            'features': list(features_info.keys()),
            'data_directory': self.robodm_data_dir,
            'sample_data_shapes': {k: v['shape'] for k, v in features_info.items()},
        }


def demo_pipeline_usage(robodm_data_dir: str):
    """Demonstrate how to use the training pipeline."""
    print("🚀 RoboDM Training Pipeline Demo")
    print("=" * 50)
    
    # Create training pipeline
    pipeline = RoboDMTrainingPipeline(robodm_data_dir)
    
    # Get training info
    training_info = pipeline.get_training_info()
    print(f"📊 Training Info:")
    for key, value in training_info.items():
        print(f"  {key}: {value}")
    
    # Get torch dataset and dataloader
    torch_dataset = pipeline.get_torch_dataset()
    dataloader = pipeline.get_dataloader(batch_size=4)
    
    print(f"\n📦 Dataset Info:")
    print(f"  Dataset size: {len(torch_dataset)}")
    print(f"  Dataloader batches: {len(dataloader)}")
    
    # Show sample batch
    sample_batch = next(iter(dataloader))
    print(f"\n🔍 Sample Batch:")
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
    
    # Get features for policy configuration
    if LEROBOT_AVAILABLE:
        try:
            policy_features = pipeline.get_policy_features()
            print(f"\n🧠 Policy Features:")
            print(f"  Input features: {list(policy_features['input_features'].keys())}")
            print(f"  Output features: {list(policy_features['output_features'].keys())}")
        except Exception as e:
            print(f"  Could not extract policy features: {e}")
    
    print(f"\n✅ Pipeline demo completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python robodm_training_pipeline.py <robodm_data_dir>")
        sys.exit(1)
    
    robodm_data_dir = sys.argv[1]
    demo_pipeline_usage(robodm_data_dir)