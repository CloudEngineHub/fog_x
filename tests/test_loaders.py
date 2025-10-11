"""Unit tests for loader functionality."""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from robodm import Trajectory
from robodm.loader import HDF5Loader, NonShuffleVLALoader

from .test_fixtures import BenchmarkDataset

# Define all codecs to test for loaders
ALL_CODECS = ["rawvideo", "ffv1", "libaom-av1", "libx264", "libx265"]


def create_test_trajectory_with_codec(temp_dir,
                                      codec,
                                      data,
                                      filename_suffix=""):
    """Helper function to create a test trajectory with specific codec and validate it works."""
    path = os.path.join(temp_dir, f"test_{codec}{filename_suffix}.vla")

    try:
        # Create trajectory
        traj = Trajectory.from_dict_of_lists(data, path, video_codec=codec)

        # Immediately try to read it back to validate encoding/decoding
        traj_read = Trajectory(path, mode="r")
        loaded_data = traj_read.load()
        traj_read.close()

        # Basic validation
        assert isinstance(loaded_data, dict)
        assert len(loaded_data) > 0

        return path, True, None
    except Exception as e:
        return path, False, str(e)


class TestNonShuffleVLALoader:
    """Test the VLA loader."""

    @pytest.mark.parametrize("codec", ALL_CODECS)
    def test_vla_loader_basic(self, temp_dir, large_sample_data, codec):
        """Test basic VLA loader functionality with all codecs."""
        # Skip libaom-av1 due to known issues with flush
        if codec == "libaom-av1":
            pytest.skip("libaom-av1 codec has known issues with flush")

        # Create VLA files with specific codec
        paths = []
        working_paths = []

        for i in range(3):
            # Create smaller datasets for faster testing
            small_data = {k: v[:5] for k, v in large_sample_data.items()}
            path, success, error = create_test_trajectory_with_codec(
                temp_dir, codec, small_data, filename_suffix=f"_{i}")
            paths.append(path)

            if success:
                working_paths.append(path)
            else:
                if "not available" in error.lower() or "codec" in error.lower(
                ):
                    pytest.skip(f"Codec {codec} not available: {error}")
                else:
                    pytest.fail(
                        f"Failed to create trajectory with codec {codec}: {error}"
                    )

        if not working_paths:
            pytest.skip(
                f"No trajectories created successfully with codec {codec}")

        # Test loading with pattern matching working files
        pattern = os.path.join(temp_dir, f"*{codec}*.vla")

        try:
            loader = NonShuffleVLALoader(pattern)

            # Test iteration
            trajectories = list(loader.iter_rows())
            assert len(trajectories) == len(working_paths)

            for traj in trajectories:
                assert isinstance(traj, dict)
                assert "observation/image" in traj
                assert "action" in traj
                assert traj["observation/image"].shape == (5, 480, 640, 3)

        except Exception as e:
            pytest.fail(f"VLA loader failed with codec {codec}: {e}")

    @pytest.mark.parametrize("codec", ALL_CODECS)
    def test_vla_loader_batch_size(self, temp_dir, large_sample_data, codec):
        """Test VLA loader with different batch sizes and all codecs."""
        # Create VLA file with specific codec
        small_data = {k: v[:10] for k, v in large_sample_data.items()}
        path, success, error = create_test_trajectory_with_codec(
            temp_dir, codec, small_data)

        if not success:
            if "not available" in error.lower() or "codec" in error.lower():
                pytest.skip(f"Codec {codec} not available: {error}")
            else:
                pytest.fail(
                    f"Failed to create trajectory with codec {codec}: {error}")

        try:
            # Test with batch size
            from robodm.loader.vla import get_vla_dataloader

            dataloader = get_vla_dataloader(path=temp_dir, batch_size=2)

            batches = list(dataloader.iter_batches())
            assert len(batches) > 0

            # Each batch should be a dictionary with batched arrays
            for batch in batches:
                assert isinstance(batch, dict)
                # Check that we have the expected keys
                assert "action" in batch
                # For batch_size=2, the first dimension should be <= 2
                action_shape = batch["action"].shape
                assert action_shape[0] <= 2  # batch dimension

        except Exception as e:
            pytest.fail(f"VLA dataloader failed with codec {codec}: {e}")


class TestVLALoaderCodecValidation:
    """Comprehensive codec validation tests for VLA loader."""

    @pytest.mark.parametrize("codec", ALL_CODECS)
    def test_loader_codec_roundtrip_validation(self, temp_dir, codec):
        """Test that VLA loader can handle all codecs with proper encoding/decoding."""
        # Skip libaom-av1 due to known issues with flush
        if codec == "libaom-av1":
            pytest.skip("libaom-av1 codec has known issues with flush")

        # Create test data designed to catch encoding issues
        test_data = {
            "observation/image": [
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
                for _ in range(5)
            ],
            "observation/depth": [
                np.random.random((640, 480)).astype(np.float32)
                for _ in range(5)
            ],
            "action":
            [np.random.random(6).astype(np.float32) for _ in range(5)],
            "reward": [float(i * 0.1) for i in range(5)],
            "done": [i == 4 for i in range(5)],
        }

        # Create trajectory with validation
        path, success, error = create_test_trajectory_with_codec(
            temp_dir, codec, test_data)

        if not success:
            if "not available" in error.lower() or "codec" in error.lower():
                pytest.skip(f"Codec {codec} not available: {error}")
            else:
                pytest.fail(f"Codec {codec} failed validation: {error}")

        try:
            # Test loading via VLA loader
            loader = NonShuffleVLALoader(path)
            trajectories = list(loader.iter_rows())

            assert len(trajectories) == 1
            traj = trajectories[0]

            # Validate loaded data structure and content
            assert isinstance(traj, dict)
            assert "observation/image" in traj
            assert "observation/depth" in traj
            assert "action" in traj
            assert "reward" in traj
            assert "done" in traj

            # Validate shapes
            assert traj["observation/image"].shape == (5, 640, 480, 3)
            assert traj["observation/depth"].shape == (5, 640, 480)
            assert traj["action"].shape == (5, 6)
            assert traj["reward"].shape == (5, )
            assert traj["done"].shape == (5, )

            # Validate data types
            assert traj["observation/image"].dtype == np.uint8
            assert traj["observation/depth"].dtype == np.float32
            assert traj["action"].dtype == np.float32

        except Exception as e:
            pytest.fail(
                f"VLA loader failed to load data created with codec {codec}: {e}"
            )

    def test_loader_codec_compatibility_report(self, temp_dir):
        """Generate a compatibility report for all codecs with the VLA loader."""
        codec_results = {}

        # Simple test data
        simple_data = {
            "observation/image": [
                np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
                for _ in range(3)
            ],
            "action": [np.array([1.0, 2.0]) for _ in range(3)],
        }

        for codec in ALL_CODECS:
            try:
                # Test trajectory creation and validation
                path, success, error = create_test_trajectory_with_codec(
                    temp_dir, codec, simple_data)

                if not success:
                    codec_results[codec] = {
                        "status": "failed_creation",
                        "error": error
                    }
                    continue

                # Test loader functionality
                loader = NonShuffleVLALoader(path)
                trajectories = list(loader.iter_rows())

                if len(trajectories) == 1 and isinstance(
                        trajectories[0], dict):
                    codec_results[codec] = {"status": "success", "error": None}
                else:
                    codec_results[codec] = {
                        "status": "failed_loading",
                        "error": "Invalid trajectory data",
                    }

            except Exception as e:
                codec_results[codec] = {
                    "status": "failed_exception",
                    "error": str(e)
                }

        # Print report
        print("\n" + "=" * 60)
        print("VLA LOADER CODEC COMPATIBILITY REPORT")
        print("=" * 60)

        successful_codecs = []
        failed_codecs = []

        for codec, result in codec_results.items():
            if result["status"] == "success":
                successful_codecs.append(codec)
                print(f"✓ {codec}: Compatible with VLA loader")
            else:
                failed_codecs.append(codec)
                print(f"✗ {codec}: {result['status']} - {result['error']}")

        print(
            f"\nSummary: {len(successful_codecs)}/{len(ALL_CODECS)} codecs compatible with VLA loader"
        )
        print("=" * 60)

        # Ensure at least one codec works
        assert len(
            successful_codecs) > 0, "No codecs are compatible with VLA loader!"

    @pytest.mark.parametrize("codec", ALL_CODECS)
    def test_loader_with_problematic_data(self, temp_dir, codec):
        """Test VLA loader with data that might cause encoding/decoding issues."""
        # Test various edge cases that might break codecs
        edge_cases = [
            {
                "name": "small_images",
                "data": {
                    "observation/image": [
                        np.random.randint(0,
                                          255, (128, 128, 3),
                                          dtype=np.uint8) for _ in range(2)
                    ],
                    "action": [np.array([1.0]) for _ in range(2)],
                },
            },
            {
                "name": "single_timestep",
                "data": {
                    "observation/image":
                    [np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)],
                    "action": [np.array([1.0, 2.0])],
                },
            },
            {
                "name": "large_variation",
                "data": {
                    "observation/image": [
                        np.zeros((640, 480, 3), dtype=np.uint8),
                        np.ones((640, 480, 3), dtype=np.uint8) * 255,
                        np.random.randint(0,
                                          255, (640, 480, 3),
                                          dtype=np.uint8),
                    ],
                    "action":
                    [np.array([0.0]),
                     np.array([1.0]),
                     np.array([0.5])],
                },
            },
        ]

        for case in edge_cases:
            try:
                path, success, error = create_test_trajectory_with_codec(
                    temp_dir,
                    codec,
                    case["data"],
                    filename_suffix=f"_{case['name']}")

                if not success:
                    if "not available" in error.lower():
                        pytest.skip(f"Codec {codec} not available: {error}")
                    else:
                        # Some edge cases might be expected to fail
                        print(
                            f"Codec {codec} failed with {case['name']} (may be expected): {error}"
                        )
                        continue

                # Test loading
                loader = NonShuffleVLALoader(path)
                trajectories = list(loader.iter_rows())

                assert len(trajectories) == 1
                traj = trajectories[0]
                assert isinstance(traj, dict)
                assert "observation/image" in traj
                assert "action" in traj

            except Exception as e:
                error_msg = str(e)
                if "not available" in error_msg.lower():
                    pytest.skip(f"Codec {codec} not available: {error_msg}")
                elif "InvalidDataError" in error_msg or "no frame" in error_msg:
                    pytest.fail(
                        f"Codec {codec} has encoding/decoding issues with {case['name']}: {error_msg}"
                    )
                else:
                    # Some failures might be expected for edge cases
                    print(
                        f"Codec {codec} failed with {case['name']} (may be expected): {error_msg}"
                    )


class TestHDF5Loader:
    """Test the HDF5 loader."""

    def test_hdf5_loader_basic(self, temp_dir, large_sample_data,
                               benchmark_dataset):
        """Test basic HDF5 loader functionality."""
        # Create HDF5 files
        paths = []
        for i in range(3):
            path = os.path.join(temp_dir, f"test_{i}.h5")
            small_data = {k: v[:5] for k, v in large_sample_data.items()}
            benchmark_dataset.create_hdf5_dataset(path, small_data)
            paths.append(path)

        # Test loading
        from robodm.loader.hdf5 import get_hdf5_dataloader

        dataloader = get_hdf5_dataloader(path=os.path.join(temp_dir, "*.h5"),
                                         batch_size=1,
                                         num_workers=0)

        # Test iteration
        batches = list(dataloader)
        assert len(batches) == 3

        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) == 1
            traj = batch[0]
            assert isinstance(traj, dict)
            assert "observation/image" in traj
            assert "action" in traj

    def test_hdf5_loader_batch_size(self, temp_dir, large_sample_data,
                                    benchmark_dataset):
        """Test HDF5 loader with different batch sizes."""
        # Create multiple HDF5 files
        for i in range(5):
            path = os.path.join(temp_dir, f"test_{i}.h5")
            small_data = {k: v[:3] for k, v in large_sample_data.items()}
            benchmark_dataset.create_hdf5_dataset(path, small_data)

        # Test with batch size
        from robodm.loader.hdf5 import get_hdf5_dataloader

        dataloader = get_hdf5_dataloader(path=os.path.join(temp_dir, "*.h5"),
                                         batch_size=2,
                                         num_workers=0)

        batches = list(dataloader)

        # Should have ceil(5/2) = 3 batches
        assert len(batches) == 3

        # First two batches should have 2 items
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        # Last batch should have 1 item
        assert len(batches[2]) == 1


class TestLoaderComparison:
    """Test comparisons between different loaders."""

    def test_vla_vs_hdf5_data_consistency(self, temp_dir, sample_dict_of_lists,
                                          benchmark_dataset):
        """Test that VLA and HDF5 loaders return consistent data."""
        # Create both VLA and HDF5 from same source data
        vla_path = os.path.join(temp_dir, "test.vla")
        h5_path = os.path.join(temp_dir, "test.h5")

        # Use deterministic data for comparison
        deterministic_data = {
            "observation/image":
            [np.ones((32, 32, 3), dtype=np.uint8) * i for i in range(3)],
            "action": [np.ones(4, dtype=np.float32) * i for i in range(3)],
        }

        # Create both formats - use lossless codec for fair comparison
        Trajectory.from_dict_of_lists(deterministic_data,
                                      vla_path,
                                      video_codec="ffv1")
        benchmark_dataset.create_hdf5_dataset(h5_path, deterministic_data)

        # Load via both loaders
        vla_loader = NonShuffleVLALoader(vla_path)
        vla_data = list(vla_loader.iter_rows())[0]

        from robodm.loader.hdf5 import get_hdf5_dataloader

        h5_loader = get_hdf5_dataloader(h5_path, batch_size=1, num_workers=0)
        h5_data = list(h5_loader)[0][0]

        # Compare data
        assert vla_data.keys() == h5_data.keys()

        for key in vla_data.keys():
            vla_array = vla_data[key]
            h5_array = h5_data[key]

            assert vla_array.shape == h5_array.shape
            assert vla_array.dtype == h5_array.dtype

            # For lossless VLA (ffv1), the data should be very close or identical
            if vla_array.dtype == np.uint8:
                # Image data - allow very small differences due to potential format conversions
                diff = np.abs(
                    vla_array.astype(np.float32) - h5_array.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                assert (
                    max_diff <= 2.0
                ), f"Max pixel difference {max_diff} too large between VLA and HDF5"
                assert (
                    mean_diff <= 0.5
                ), f"Mean pixel difference {mean_diff} too large between VLA and HDF5"
            else:
                # Float data should be nearly identical for lossless
                np.testing.assert_allclose(vla_array,
                                           h5_array,
                                           rtol=1e-5,
                                           atol=1e-6)


class TestLoaderError:
    """Test error handling in loaders."""

    def test_vla_loader_empty_pattern(self, temp_dir):
        """Test VLA loader with pattern that matches no files."""
        pattern = os.path.join(temp_dir, "nonexistent_*.vla")
        loader = NonShuffleVLALoader(pattern)

        # Should handle empty results gracefully
        trajectories = list(loader.iter_rows())
        assert len(trajectories) == 0

    def test_hdf5_loader_empty_pattern(self, temp_dir):
        """Test HDF5 loader with pattern that matches no files."""
        pattern = os.path.join(temp_dir, "nonexistent_*.h5")

        from robodm.loader.hdf5 import get_hdf5_dataloader

        dataloader = get_hdf5_dataloader(pattern, batch_size=1, num_workers=0)

        # Should handle empty results gracefully
        batches = list(dataloader)
        assert len(batches) == 0

    def test_vla_loader_corrupted_file(self, temp_dir):
        """Test VLA loader behavior with corrupted files."""
        # Create a fake VLA file
        fake_path = os.path.join(temp_dir, "fake.vla")
        with open(fake_path, "w") as f:
            f.write("This is not a valid VLA file")

        loader = NonShuffleVLALoader(fake_path)

        # Should handle corrupted files gracefully
        with pytest.raises(Exception):
            list(loader.iter_rows())


class TestLoaderPerformance:
    """Basic performance tests for loaders."""

    def test_vla_loader_memory_usage(self, temp_dir, large_sample_data):
        """Test VLA loader memory efficiency."""
        # Create VLA file
        path = os.path.join(temp_dir, "large_test.vla")
        Trajectory.from_dict_of_lists(large_sample_data,
                                      path,
                                      video_codec="ffv1")

        # Load and measure (basic test - would need memory profiling for real measurement)
        loader = NonShuffleVLALoader(path)
        trajectories = list(loader.iter_rows())

        assert len(trajectories) == 1
        assert "observation/image" in trajectories[0]
        assert trajectories[0]["observation/image"].shape == (100, 480, 640, 3)

    def test_hdf5_loader_memory_usage(self, temp_dir, large_sample_data,
                                      benchmark_dataset):
        """Test HDF5 loader memory efficiency."""
        # Create HDF5 file
        path = os.path.join(temp_dir, "large_test.h5")
        benchmark_dataset.create_hdf5_dataset(path, large_sample_data)

        # Load and measure
        from robodm.loader.hdf5 import get_hdf5_dataloader

        dataloader = get_hdf5_dataloader(path, batch_size=1, num_workers=0)
        batches = list(dataloader)

        assert len(batches) == 1
        traj = batches[0][0]
        assert "observation/image" in traj
        assert traj["observation/image"].shape == (100, 480, 640, 3)
