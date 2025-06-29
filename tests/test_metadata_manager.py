"""Tests for the MetadataManager system."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robodm.metadata_manager import MetadataManager, TrajectoryMetadata


@pytest.fixture
def sample_trajectory_metadata():
    """Create sample trajectory metadata."""
    return [
        TrajectoryMetadata(
            file_path="/path/to/traj1.vla",
            trajectory_length=100,
            feature_keys=["action", "observation/images/cam_high"],
            feature_shapes={
                "action": [7],
                "observation/images/cam_high": [128, 128, 3],
            },
            feature_dtypes={
                "action": "float32",
                "observation/images/cam_high": "uint8",
            },
            file_size=1024000,
            last_modified=datetime(2023, 1, 1, 12, 0, 0),
            checksum="abc123",
        ),
        TrajectoryMetadata(
            file_path="/path/to/traj2.vla",
            trajectory_length=150,
            feature_keys=["action", "observation/state/joint_pos"],
            feature_shapes={
                "action": [7],
                "observation/state/joint_pos": [7]
            },
            feature_dtypes={
                "action": "float32",
                "observation/state/joint_pos": "float32",
            },
            file_size=2048000,
            last_modified=datetime(2023, 1, 2, 12, 0, 0),
            checksum="def456",
        ),
    ]


@pytest.fixture
def temp_dataset_dir(temp_dir):
    """Create a temporary dataset directory."""
    dataset_dir = temp_dir / "test_dataset"
    dataset_dir.mkdir()
    return dataset_dir


class TestTrajectoryMetadata:
    """Test TrajectoryMetadata class."""

    def test_to_dict(self):
        """Test converting TrajectoryMetadata to dictionary."""
        metadata = TrajectoryMetadata(
            file_path="/test/path.vla",
            trajectory_length=100,
            feature_keys=["action"],
            feature_shapes={"action": [7]},
            feature_dtypes={"action": "float32"},
            file_size=1024,
            last_modified=datetime(2023, 1, 1, 12, 0, 0),
            checksum="abc123",
        )

        result = metadata.to_dict()

        assert result["file_path"] == "/test/path.vla"
        assert result["trajectory_length"] == 100
        assert result["feature_keys"] == ["action"]
        assert result["feature_shapes"] == {"action": [7]}
        assert result["feature_dtypes"] == {"action": "float32"}
        assert result["file_size"] == 1024
        assert result["last_modified"] == "2023-01-01T12:00:00"
        assert result["checksum"] == "abc123"

    def test_from_dict(self):
        """Test creating TrajectoryMetadata from dictionary."""
        data = {
            "file_path": "/test/path.vla",
            "trajectory_length": 100,
            "feature_keys": ["action"],
            "feature_shapes": {
                "action": [7]
            },
            "feature_dtypes": {
                "action": "float32"
            },
            "file_size": 1024,
            "last_modified": "2023-01-01T12:00:00",
            "checksum": "abc123",
        }

        metadata = TrajectoryMetadata.from_dict(data)

        assert metadata.file_path == "/test/path.vla"
        assert metadata.trajectory_length == 100
        assert metadata.feature_keys == ["action"]
        assert metadata.feature_shapes == {"action": [7]}
        assert metadata.feature_dtypes == {"action": "float32"}
        assert metadata.file_size == 1024
        assert metadata.last_modified == datetime(2023, 1, 1, 12, 0, 0)
        assert metadata.checksum == "abc123"

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion to_dict -> from_dict."""
        original = TrajectoryMetadata(
            file_path="/test/path.vla",
            trajectory_length=100,
            feature_keys=["action", "observation"],
            feature_shapes={
                "action": [7],
                "observation": [128, 128, 3]
            },
            feature_dtypes={
                "action": "float32",
                "observation": "uint8"
            },
            file_size=1024,
            last_modified=datetime(2023, 1, 1, 12, 0, 0),
        )

        dict_data = original.to_dict()
        reconstructed = TrajectoryMetadata.from_dict(dict_data)

        assert reconstructed.file_path == original.file_path
        assert reconstructed.trajectory_length == original.trajectory_length
        assert reconstructed.feature_keys == original.feature_keys
        assert reconstructed.feature_shapes == original.feature_shapes
        assert reconstructed.feature_dtypes == original.feature_dtypes
        assert reconstructed.file_size == original.file_size
        assert reconstructed.last_modified == original.last_modified
        assert reconstructed.checksum == original.checksum


class TestMetadataManager:
    """Test MetadataManager class."""

    def test_init(self, temp_dataset_dir):
        """Test MetadataManager initialization."""
        manager = MetadataManager(temp_dataset_dir)

        assert manager.dataset_path == temp_dataset_dir
        assert manager.metadata_path == temp_dataset_dir / "trajectory_metadata.parquet"
        assert manager._metadata_cache is None

    def test_init_custom_filename(self, temp_dataset_dir):
        """Test MetadataManager initialization with custom filename."""
        manager = MetadataManager(temp_dataset_dir, "custom_metadata.parquet")

        assert manager.metadata_path == temp_dataset_dir / "custom_metadata.parquet"

    def test_exists_false(self, temp_dataset_dir):
        """Test exists() when metadata file doesn't exist."""
        manager = MetadataManager(temp_dataset_dir)
        assert not manager.exists()

    def test_exists_true(self, temp_dataset_dir, sample_trajectory_metadata):
        """Test exists() when metadata file exists."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        assert manager.exists()

    def test_save_metadata(self, temp_dataset_dir, sample_trajectory_metadata):
        """Test saving metadata to parquet file."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        assert manager.metadata_path.exists()

        # Verify parquet file content
        df = pd.read_parquet(manager.metadata_path)
        assert len(df) == 2
        assert list(df.columns) == [
            "file_path",
            "trajectory_length",
            "feature_keys",
            "feature_shapes",
            "feature_dtypes",
            "file_size",
            "last_modified",
            "checksum",
        ]
        assert df.iloc[0]["file_path"] == "/path/to/traj1.vla"
        assert df.iloc[0]["trajectory_length"] == 100
        assert df.iloc[1]["trajectory_length"] == 150

    def test_save_metadata_empty_list(self, temp_dataset_dir):
        """Test saving empty metadata list."""
        manager = MetadataManager(temp_dataset_dir)

        with patch("robodm.metadata_manager.logger") as mock_logger:
            manager.save_metadata([])
            mock_logger.warning.assert_called_once_with("No metadata to save")

        assert not manager.metadata_path.exists()

    def test_save_metadata_exception_handling(self, temp_dataset_dir,
                                              sample_trajectory_metadata):
        """Test exception handling during save."""
        manager = MetadataManager(temp_dataset_dir)

        with patch("pandas.DataFrame.to_parquet",
                   side_effect=Exception("Save failed")):
            with patch("robodm.metadata_manager.logger") as mock_logger:
                with pytest.raises(Exception, match="Save failed"):
                    manager.save_metadata(sample_trajectory_metadata)

                mock_logger.error.assert_called_once()

    def test_load_metadata(self, temp_dataset_dir, sample_trajectory_metadata):
        """Test loading metadata from parquet file."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        df = manager.load_metadata()

        assert len(df) == 2
        assert df.iloc[0]["file_path"] == "/path/to/traj1.vla"
        assert df.iloc[1]["file_path"] == "/path/to/traj2.vla"
        assert manager._metadata_cache is not None

    def test_load_metadata_caching(self, temp_dataset_dir,
                                   sample_trajectory_metadata):
        """Test metadata caching functionality."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        # First load
        df1 = manager.load_metadata()

        # Second load should use cache
        with patch("pandas.read_parquet") as mock_read:
            df2 = manager.load_metadata()
            mock_read.assert_not_called()

        assert df1 is df2

    def test_load_metadata_force_reload(self, temp_dataset_dir,
                                        sample_trajectory_metadata):
        """Test forcing metadata reload."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        # First load
        manager.load_metadata()

        # Force reload should bypass cache
        with patch("pandas.read_parquet",
                   return_value=pd.DataFrame()) as mock_read:
            manager.load_metadata(force_reload=True)
            mock_read.assert_called_once()

    def test_load_metadata_file_not_found(self, temp_dataset_dir):
        """Test loading metadata when file doesn't exist."""
        manager = MetadataManager(temp_dataset_dir)

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            manager.load_metadata()

    def test_load_metadata_exception_handling(self, temp_dataset_dir):
        """Test exception handling during load."""
        manager = MetadataManager(temp_dataset_dir)
        # Create an invalid parquet file
        manager.metadata_path.write_text("invalid parquet content")

        with patch("robodm.metadata_manager.logger") as mock_logger:
            with pytest.raises(Exception):
                manager.load_metadata()

            mock_logger.error.assert_called_once()

    def test_get_trajectory_metadata(self, temp_dataset_dir,
                                     sample_trajectory_metadata):
        """Test getting metadata for specific trajectory."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        metadata = manager.get_trajectory_metadata("/path/to/traj1.vla")

        assert metadata is not None
        assert metadata.file_path == "/path/to/traj1.vla"
        assert metadata.trajectory_length == 100
        assert metadata.checksum == "abc123"

    def test_get_trajectory_metadata_not_found(self, temp_dataset_dir,
                                               sample_trajectory_metadata):
        """Test getting metadata for non-existent trajectory."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        metadata = manager.get_trajectory_metadata("/path/to/nonexistent.vla")

        assert metadata is None

    def test_get_trajectory_metadata_path_normalization(
            self, temp_dataset_dir, sample_trajectory_metadata):
        """Test path normalization in get_trajectory_metadata."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        with patch("pathlib.Path.resolve",
                   return_value=Path("/path/to/traj1.vla")):
            metadata = manager.get_trajectory_metadata("../path/to/traj1.vla")
            assert metadata is not None

    def test_update_metadata_no_existing(self, temp_dataset_dir,
                                         sample_trajectory_metadata):
        """Test updating metadata when no existing file."""
        manager = MetadataManager(temp_dataset_dir)

        manager.update_metadata(sample_trajectory_metadata[:1])

        assert manager.exists()
        df = manager.load_metadata(force_reload=True)
        assert len(df) == 1

    def test_update_metadata_existing_file(self, temp_dataset_dir,
                                           sample_trajectory_metadata):
        """Test updating existing metadata."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        # Update first trajectory with new length
        updated_metadata = TrajectoryMetadata(
            file_path="/path/to/traj1.vla",
            trajectory_length=200,  # Changed from 100
            feature_keys=["action", "observation/images/cam_high"],
            feature_shapes={
                "action": [7],
                "observation/images/cam_high": [128, 128, 3],
            },
            feature_dtypes={
                "action": "float32",
                "observation/images/cam_high": "uint8",
            },
            file_size=2048000,  # Changed from 1024000
            last_modified=datetime(2023, 1, 15, 12, 0, 0),
            checksum="updated123",
        )

        manager.update_metadata([updated_metadata])

        df = manager.load_metadata(force_reload=True)
        assert len(df) == 2  # Still 2 trajectories

        # Check that first trajectory was updated
        traj1_row = df[df["file_path"] == "/path/to/traj1.vla"].iloc[0]
        assert traj1_row["trajectory_length"] == 200
        assert traj1_row["file_size"] == 2048000
        assert traj1_row["checksum"] == "updated123"

        # Check that second trajectory is unchanged
        traj2_row = df[df["file_path"] == "/path/to/traj2.vla"].iloc[0]
        assert traj2_row["trajectory_length"] == 150

    def test_update_metadata_add_new_trajectories(self, temp_dataset_dir,
                                                  sample_trajectory_metadata):
        """Test adding new trajectories to existing metadata."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(
            sample_trajectory_metadata[:1])  # Save only first trajectory

        new_metadata = TrajectoryMetadata(
            file_path="/path/to/traj3.vla",
            trajectory_length=75,
            feature_keys=["action"],
            feature_shapes={"action": [7]},
            feature_dtypes={"action": "float32"},
            file_size=512000,
            last_modified=datetime(2023, 1, 3, 12, 0, 0),
            checksum="new789",
        )

        manager.update_metadata([new_metadata])

        df = manager.load_metadata(force_reload=True)
        assert len(df) == 2  # Original + new trajectory

        # Check new trajectory was added
        new_row = df[df["file_path"] == "/path/to/traj3.vla"].iloc[0]
        assert new_row["trajectory_length"] == 75
        assert new_row["checksum"] == "new789"

    def test_remove_metadata(self, temp_dataset_dir,
                             sample_trajectory_metadata):
        """Test removing metadata for specific trajectories."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        manager.remove_metadata(["/path/to/traj1.vla"])

        df = manager.load_metadata(force_reload=True)
        assert len(df) == 1
        assert df.iloc[0]["file_path"] == "/path/to/traj2.vla"

    def test_remove_metadata_no_file(self, temp_dataset_dir):
        """Test removing metadata when no file exists."""
        manager = MetadataManager(temp_dataset_dir)

        with patch("robodm.metadata_manager.logger") as mock_logger:
            manager.remove_metadata(["/path/to/traj1.vla"])
            mock_logger.warning.assert_called_once_with(
                "No metadata file to remove from")

    def test_remove_metadata_path_normalization(self, temp_dataset_dir,
                                                sample_trajectory_metadata):
        """Test path normalization in remove_metadata."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        with patch("pathlib.Path.resolve",
                   return_value=Path("/path/to/traj1.vla")):
            manager.remove_metadata(["../path/to/traj1.vla"])

        df = manager.load_metadata(force_reload=True)
        assert len(df) == 1
        assert df.iloc[0]["file_path"] == "/path/to/traj2.vla"

    def test_get_all_metadata(self, temp_dataset_dir,
                              sample_trajectory_metadata):
        """Test getting all trajectory metadata."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        all_metadata = manager.get_all_metadata()

        assert len(all_metadata) == 2
        assert all(
            isinstance(meta, TrajectoryMetadata) for meta in all_metadata)
        assert all_metadata[0].file_path == "/path/to/traj1.vla"
        assert all_metadata[1].file_path == "/path/to/traj2.vla"

    def test_filter_by_length(self, temp_dataset_dir,
                              sample_trajectory_metadata):
        """Test filtering trajectories by length."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        # Test min_length filter
        long_trajs = manager.filter_by_length(min_length=120)
        assert len(long_trajs) == 1
        assert long_trajs[0].trajectory_length == 150

        # Test max_length filter
        short_trajs = manager.filter_by_length(max_length=120)
        assert len(short_trajs) == 1
        assert short_trajs[0].trajectory_length == 100

        # Test both filters
        medium_trajs = manager.filter_by_length(min_length=50, max_length=120)
        assert len(medium_trajs) == 1
        assert medium_trajs[0].trajectory_length == 100

        # Test no matches
        no_matches = manager.filter_by_length(min_length=200)
        assert len(no_matches) == 0

    def test_get_statistics(self, temp_dataset_dir,
                            sample_trajectory_metadata):
        """Test getting dataset statistics."""
        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata(sample_trajectory_metadata)

        stats = manager.get_statistics()

        expected_stats = {
            "total_trajectories": 2,
            "total_timesteps": 250,  # 100 + 150
            "average_length": 125.0,  # (100 + 150) / 2
            "min_length": 100,
            "max_length": 150,
            "total_size_bytes": 3072000,  # 1024000 + 2048000
            "unique_feature_keys": {
                "action",
                "observation/images/cam_high",
                "observation/state/joint_pos",
            },
        }

        assert stats["total_trajectories"] == expected_stats[
            "total_trajectories"]
        assert stats["total_timesteps"] == expected_stats["total_timesteps"]
        assert stats["average_length"] == expected_stats["average_length"]
        assert stats["min_length"] == expected_stats["min_length"]
        assert stats["max_length"] == expected_stats["max_length"]
        assert stats["total_size_bytes"] == expected_stats["total_size_bytes"]
        assert (set(stats["unique_feature_keys"]) ==
                expected_stats["unique_feature_keys"])

    def test_get_statistics_empty_dataset(self, temp_dataset_dir):
        """Test getting statistics for empty dataset."""
        # Create empty parquet file
        manager = MetadataManager(temp_dataset_dir)
        empty_df = pd.DataFrame(columns=[
            "file_path",
            "trajectory_length",
            "feature_keys",
            "feature_shapes",
            "feature_dtypes",
            "file_size",
            "last_modified",
            "checksum",
        ])
        empty_df.to_parquet(manager.metadata_path, index=False)

        stats = manager.get_statistics()

        assert stats["total_trajectories"] == 0
        assert stats["total_timesteps"] == 0
        assert stats["unique_feature_keys"] == []

    def test_get_statistics_malformed_feature_keys(self, temp_dataset_dir):
        """Test getting statistics with malformed feature_keys."""
        manager = MetadataManager(temp_dataset_dir)

        # Create DataFrame with mixed feature_keys types
        df = pd.DataFrame({
            "file_path": ["/path/traj1.vla", "/path/traj2.vla"],
            "trajectory_length": [100, 150],
            "feature_keys": [["action"], "not_a_list"],  # Mixed types
            "feature_shapes": [{}, {}],
            "feature_dtypes": [{}, {}],
            "file_size": [1000, 2000],
            "last_modified": ["2023-01-01T12:00:00", "2023-01-02T12:00:00"],
            "checksum": ["abc", "def"],
        })
        df.to_parquet(manager.metadata_path, index=False)

        stats = manager.get_statistics()

        # Should handle non-list feature_keys gracefully
        assert stats["total_trajectories"] == 2
        assert "action" in stats["unique_feature_keys"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_metadata_manager_with_string_path(self, temp_dir):
        """Test MetadataManager with string path instead of Path object."""
        manager = MetadataManager(str(temp_dir))
        assert isinstance(manager.dataset_path, Path)
        assert manager.dataset_path == temp_dir

    def test_concurrent_access_simulation(self, temp_dataset_dir,
                                          sample_trajectory_metadata):
        """Test handling of concurrent access scenarios."""
        manager1 = MetadataManager(temp_dataset_dir)
        manager2 = MetadataManager(temp_dataset_dir)

        # Manager 1 saves metadata
        manager1.save_metadata(sample_trajectory_metadata[:1])

        # Manager 2 loads (should work)
        df = manager2.load_metadata()
        assert len(df) == 1

        # Manager 1 adds more metadata
        manager1.update_metadata(sample_trajectory_metadata[1:])

        # Manager 2 force reload to see updates
        df = manager2.load_metadata(force_reload=True)
        assert len(df) == 2

    def test_very_long_file_paths(self, temp_dataset_dir):
        """Test handling of very long file paths."""
        long_path = "/very/long/path/" + "subdir/" * 50 + "trajectory.vla"

        metadata = TrajectoryMetadata(
            file_path=long_path,
            trajectory_length=100,
            feature_keys=["action"],
            feature_shapes={"action": [7]},
            feature_dtypes={"action": "float32"},
            file_size=1024,
            last_modified=datetime.now(),
        )

        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata([metadata])

        retrieved = manager.get_trajectory_metadata(long_path)
        assert retrieved is not None
        assert retrieved.file_path == long_path

    def test_special_characters_in_paths(self, temp_dataset_dir):
        """Test handling of special characters in file paths."""
        special_path = "/path/with spaces/and-dashes/traj_with_ünïcödë.vla"

        metadata = TrajectoryMetadata(
            file_path=special_path,
            trajectory_length=100,
            feature_keys=["action"],
            feature_shapes={"action": [7]},
            feature_dtypes={"action": "float32"},
            file_size=1024,
            last_modified=datetime.now(),
        )

        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata([metadata])

        retrieved = manager.get_trajectory_metadata(special_path)
        assert retrieved is not None
        assert retrieved.file_path == special_path

    def test_large_feature_shapes(self, temp_dataset_dir):
        """Test handling of large and complex feature shapes."""
        complex_shapes = {
            "observation/images/cam1": [480, 640, 3],
            "observation/images/cam2": [480, 640, 3],
            "observation/images/cam3": [480, 640, 3],
            "observation/pointcloud": [1000000, 3],
            "action": [50],  # High-dimensional action space
            "observation/proprioception": [100],
        }

        metadata = TrajectoryMetadata(
            file_path="/path/to/complex_traj.vla",
            trajectory_length=1000,
            feature_keys=list(complex_shapes.keys()),
            feature_shapes=complex_shapes,
            feature_dtypes={k: "float32"
                            for k in complex_shapes.keys()},
            file_size=10**9,  # 1GB file
            last_modified=datetime.now(),
        )

        manager = MetadataManager(temp_dataset_dir)
        manager.save_metadata([metadata])

        retrieved = manager.get_trajectory_metadata(
            "/path/to/complex_traj.vla")
        assert retrieved is not None
        assert retrieved.feature_shapes == complex_shapes
        assert len(retrieved.feature_keys) == 6
