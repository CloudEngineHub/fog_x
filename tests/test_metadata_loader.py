#!/usr/bin/env python3
"""Test script for the metadata-enhanced VLA loader."""

import logging
import os
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import robodm
from robodm.loader.vla import LoadingMode, RayVLALoader, SliceConfig
from robodm.metadata_manager import MetadataManager
from robodm.metadata_utils import build_dataset_metadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_trajectories(temp_dir: Path, num_trajectories: int = 3):
    """Create some test trajectory files."""
    logger.info(f"Creating {num_trajectories} test trajectories in {temp_dir}")

    trajectory_files = []
    for i in range(num_trajectories):
        # Create trajectory with varying lengths
        traj_length = 100 + i * 50  # 100, 150, 200

        # Create sample data
        observations_image = np.random.randint(0,
                                               255, (traj_length, 640, 480, 3),
                                               dtype=np.uint8)
        observations_state = np.random.randn(traj_length, 7).astype(np.float32)
        actions = np.random.randn(traj_length, 7).astype(np.float32)

        # Save trajectory
        traj_file = temp_dir / f"trajectory_{i}.vla"
        traj = robodm.Trajectory(str(traj_file), mode="w")

        # Add data for each timestep
        for t in range(traj_length):
            timestep_data = {
                "observations": {
                    "image": observations_image[t],
                    "state": observations_state[t],
                },
                "actions": actions[t],
                "metadata": {
                    "episode_id": f"episode_{i}",
                    "robot_name": "test_robot",
                    "timestep": t,
                },
            }
            traj.add_by_dict(timestep_data)

        traj.close()

        trajectory_files.append(traj_file)
        logger.info(f"Created trajectory {i} with length {traj_length}")

    return trajectory_files


def test_metadata_loading():
    """Test the metadata-enhanced loader."""
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test trajectories
        trajectory_files = create_test_trajectories(temp_path)

        logger.info("\n=== Testing without metadata (first run) ===")
        # First run - metadata will be built automatically
        start_time = time.time()
        loader1 = RayVLALoader(
            path=str(temp_path / "*.vla"),
            mode=LoadingMode.TRAJECTORY,
            use_metadata=True,
            auto_build_metadata=True,
        )

        # Count trajectories
        count1 = loader1.count()
        logger.info(f"Found {count1} trajectories")
        logger.info(f"Time to initialize: {time.time() - start_time:.2f}s")

        # Check that metadata was created
        metadata_manager = MetadataManager(temp_path)
        assert metadata_manager.exists(
        ), "Metadata file should have been created"

        # Get statistics
        stats = metadata_manager.get_statistics()
        logger.info(f"Dataset statistics: {stats}")

        logger.info("\n=== Testing with existing metadata (second run) ===")
        # Second run - should use existing metadata
        start_time = time.time()
        loader2 = RayVLALoader(
            path=str(temp_path / "*.vla"),
            mode=LoadingMode.TRAJECTORY,
            use_metadata=True,
            auto_build_metadata=False,  # Won't build if missing
        )

        count2 = loader2.count()
        logger.info(f"Found {count2} trajectories")
        logger.info(f"Time to initialize: {time.time() - start_time:.2f}s")

        assert count1 == count2, "Should find same number of trajectories"

        logger.info("\n=== Testing slice mode with metadata ===")
        # Test slice mode
        loader3 = RayVLALoader(
            path=str(temp_path / "*.vla"),
            mode=LoadingMode.SLICE,
            slice_config=SliceConfig(slice_length=50, min_slice_length=30),
            use_metadata=True,
        )

        # Take a few slices
        slices = loader3.take(5)
        logger.info(f"Got {len(slices)} slices")
        if slices:
            first_slice = slices[0]
            logger.info(f"First slice keys: {list(first_slice.keys())}")
            if "actions" in first_slice:
                logger.info(
                    f"First slice action shape: {first_slice['actions'].shape}"
                )

        logger.info("\n=== Testing metadata filtering ===")
        # Test filtering by length
        long_trajectories = metadata_manager.filter_by_length(min_length=150)
        logger.info(
            f"Found {len(long_trajectories)} trajectories with length >= 150")

        for meta in long_trajectories:
            logger.info(
                f"  - {Path(meta.file_path).name}: length={meta.trajectory_length}"
            )

        logger.info("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    test_metadata_loading()
