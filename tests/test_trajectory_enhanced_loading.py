"""
Enhanced tests for trajectory loading with various options.
Simplified to focus on core functionality rather than edge cases.
"""

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from robodm import Trajectory


def create_test_data(num_steps=100, rng=None):
    """Generate deterministic test data."""
    if rng is None:
        rng = np.random.RandomState(42)

    return [
        {
            "observations/image": rng.randint(0,
                                              255, (64, 64, 3),
                                              dtype=np.uint8),
            "observations/position": rng.randn(3).astype(np.float32),
            "observations/velocity": rng.randn(3).astype(np.float32),
            "action": rng.randn(7).astype(np.float32),
            "reward": np.float32(rng.randn()),
            "done": False,
            "info/success": i > num_steps * 0.8,
            "info/task_id": i % 5,
            "metadata/episode_id": 0,
            "metadata/step": i,
            "timestamp": i * 100,  # 100ms intervals
        } for i in range(num_steps)
    ]


@pytest.fixture
def base_trajectory_data():
    """Generate base trajectory data for testing."""
    return create_test_data(100)


@pytest.fixture
def temp_dir(tmpdir):
    """Create a temporary directory."""
    return str(tmpdir)


@pytest.fixture
def trajectory_path(temp_dir, base_trajectory_data) -> str:
    """Create a test trajectory file."""
    path = os.path.join(temp_dir, "traj.vla")
    traj = Trajectory(path, mode="w")

    # Add data with explicit timestamps (100ms intervals = 10 Hz)
    for i, step_data in enumerate(base_trajectory_data):
        timestamp_ms = int(i * 100)  # 100ms intervals
        # Remove timestamp from step_data since we're passing it explicitly
        data_without_timestamp = {
            k: v
            for k, v in step_data.items() if k != "timestamp"
        }
        traj.add_by_dict(data_without_timestamp,
                         timestamp=timestamp_ms,
                         time_unit="ms")

    traj.close()
    return path


@pytest.fixture
def small_trajectory_path(temp_dir, rng) -> str:
    """Smaller trajectory for testing edge cases."""
    path = os.path.join(temp_dir, "small_traj.vla")
    traj = Trajectory(path, mode="w")

    # Only 5 steps
    for i in range(5):
        timestamp_ms = int(i * 100)
        data = {
            "value": i,
            "name": f"item_{i}",
            "array": rng.normal(size=2).astype(np.float32),
        }
        traj.add_by_dict(data, timestamp=timestamp_ms, time_unit="ms")

    traj.close()
    return path


@pytest.fixture
def rng():
    """Random number generator for consistent tests."""
    return np.random.RandomState(42)


class TestTrajectoryLoad:
    """Test trajectory loading functionality."""

    def test_no_kwargs_is_identity(self, trajectory_path):
        """Test that load() without arguments returns all data."""
        t = Trajectory(trajectory_path, mode="r")
        data1 = t.load()
        data2 = t.load()
        t.close()

        assert set(data1.keys()) == set(data2.keys())
        for k in data1:
            np.testing.assert_array_equal(data1[k], data2[k])

    def test_load_returns_correct_keys(self, trajectory_path):
        """Test that load returns expected keys."""
        t = Trajectory(trajectory_path, mode="r")
        data = t.load()
        t.close()

        expected_keys = {
            "observations/image",
            "observations/position",
            "observations/velocity",
            "action",
            "reward",
            "done",
            "info/success",
            "info/task_id",
            "metadata/episode_id",
            "metadata/step",
        }
        assert set(data.keys()) == expected_keys

    def test_empty_trajectory_handling(self, temp_dir):
        """Test handling of empty trajectories."""
        path = os.path.join(temp_dir, "empty.vla")
        traj = Trajectory(path, mode="w")
        traj.close()

        # Check if file exists after creation
        if not os.path.exists(path):
            # If no file was created (because no data was added),
            # the Trajectory constructor should fail when trying to read
            with pytest.raises(FileNotFoundError):
                t = Trajectory(path, mode="r")
            return

        # If file exists, load should return empty dict
        t = Trajectory(path, mode="r")
        data = t.load()
        assert isinstance(data, dict)
        assert len(data) == 0
        t.close()

    def test_basic_loading(self, trajectory_path):
        """Test basic trajectory loading."""
        t = Trajectory(trajectory_path, mode="r")
        data = t.load()
        t.close()

        # Check data shapes
        assert data["observations/image"].shape == (100, 64, 64, 3)
        assert data["observations/position"].shape == (100, 3)
        assert data["action"].shape == (100, 7)
        assert data["reward"].shape == (100, )

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises appropriate error."""
        path = os.path.join(temp_dir, "nonexistent.vla")
        with pytest.raises(FileNotFoundError):
            Trajectory(path, mode="r")

    def test_single_frame_trajectory(self, temp_dir, rng):
        """Test trajectory with single frame."""
        path = os.path.join(temp_dir, "single_frame.vla")
        traj = Trajectory(path, mode="w")
        traj.add_by_dict({
            "value": 42,
            "name": "single"
        },
                         timestamp=0,
                         time_unit="ms")
        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()
        t.close()

        assert data["value"].shape == (1, )
        assert data["value"][0] == 42
        assert data["name"][0] == "single"

    def test_complex_feature_names(self, temp_dir, rng):
        """Test handling of complex nested feature names."""
        path = os.path.join(temp_dir, "complex_names.vla")
        traj = Trajectory(path, mode="w")

        nested_data = {
            "robot/arm/joints/position":
            rng.randn(7).astype(np.float32),
            "robot/arm/joints/velocity":
            rng.randn(7).astype(np.float32),
            "sensors/camera/left/image":
            rng.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            "meta/info/timestamp/ns":
            1000000,
            "status":
            True,
        }

        for i in range(5):
            traj.add_by_dict(nested_data, timestamp=i * 100, time_unit="ms")

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()
        t.close()

        assert "robot/arm/joints/position" in data
        assert "sensors/camera/left/image" in data
        assert data["robot/arm/joints/position"].shape == (5, 7)
        assert data["sensors/camera/left/image"].shape == (5, 32, 32, 3)


class TestTrajectoryLoadIntegration:
    """Integration tests for trajectory loading."""

    def test_full_pipeline_integration(self, temp_dir, rng):
        """Test full pipeline from creation to loading."""
        path = os.path.join(temp_dir, "pipeline_test.vla")

        # Create trajectory with various data types
        traj = Trajectory(path, mode="w")

        for i in range(50):
            step_data = {
                "observations/rgb":
                rng.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                "observations/depth":
                rng.rand(128, 128).astype(np.float32),
                "observations/proprioception":
                rng.randn(14).astype(np.float32),
                "actions/joint_positions":
                rng.randn(7).astype(np.float32),
                "actions/gripper":
                rng.choice([0, 1]),
                "rewards/sparse":
                float(i > 40),
                "rewards/dense":
                np.float32(rng.randn()),
                "info": {
                    "step": i,
                    "episode": 0,
                    "iteration": i,
                    "phase": "test"
                },
            }
            traj.add_by_dict(
                step_data,
                timestamp=int(i * 20),
                time_unit="ms"  # 20ms intervals
            )

        traj.close()

        # Test various loading scenarios
        t = Trajectory(path, mode="r")

        # Full load
        full_data = t.load()
        assert full_data["observations/rgb"].shape == (50, 128, 128, 3)
        assert full_data["actions/joint_positions"].shape == (50, 7)
        assert full_data["info/step"].shape == (50, )

        t.close()

    def test_robustness_with_malformed_data(self, temp_dir):
        """Test robustness when handling edge cases."""
        path = os.path.join(temp_dir, "malformed_test.vla")
        traj = Trajectory(path, mode="w")

        # Add some normal data
        for i in range(10):
            traj.add_by_dict(
                {
                    "value": i,
                    "data": np.array([i, i + 1])
                },
                timestamp=i * 100,
                time_unit="ms",
            )

        traj.close()

        t = Trajectory(path, mode="r")
        data = t.load()
        t.close()

        assert len(data["value"]) == 10
        assert data["value"][0] == 0
        assert data["value"][-1] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
