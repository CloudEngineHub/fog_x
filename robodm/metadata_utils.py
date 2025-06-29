import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import robodm
from robodm.metadata_manager import MetadataManager, TrajectoryMetadata

logger = logging.getLogger(__name__)


def compute_file_checksum(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_trajectory_metadata(file_path: str,
                                compute_checksum: bool = False
                                ) -> TrajectoryMetadata:
    """
    Extract metadata from a trajectory file.

    Args:
        file_path: Path to the trajectory file
        compute_checksum: Whether to compute file checksum (slower but ensures data integrity)

    Returns:
        TrajectoryMetadata object
    """
    file_path = str(Path(file_path).resolve())

    try:
        # Load trajectory to extract metadata
        traj = robodm.Trajectory(file_path)
        data = traj.load(return_type="numpy")

        if not data:
            raise ValueError(f"Empty trajectory data in {file_path}")

        # Extract trajectory length from first feature
        first_key = next(iter(data.keys()))
        trajectory_length = len(data[first_key])

        # Extract feature information
        feature_keys = list(data.keys())
        feature_shapes = {}
        feature_dtypes = {}

        for key, value in data.items():
            if hasattr(value, "shape"):
                # For numpy arrays
                feature_shapes[key] = list(
                    value.shape[1:])  # Exclude time dimension
                feature_dtypes[key] = str(value.dtype)
            elif isinstance(value, list) and len(value) > 0:
                # For lists
                if hasattr(value[0], "shape"):
                    feature_shapes[key] = list(value[0].shape)
                    feature_dtypes[key] = str(value[0].dtype)
                else:
                    feature_shapes[key] = []
                    feature_dtypes[key] = type(value[0]).__name__
            else:
                feature_shapes[key] = []
                feature_dtypes[key] = type(value).__name__

        # Get file metadata
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        last_modified = datetime.fromtimestamp(file_stat.st_mtime)

        # Compute checksum if requested
        checksum = None
        if compute_checksum:
            checksum = compute_file_checksum(file_path)

        return TrajectoryMetadata(
            file_path=file_path,
            trajectory_length=trajectory_length,
            feature_keys=feature_keys,
            feature_shapes=feature_shapes,
            feature_dtypes=feature_dtypes,
            file_size=file_size,
            last_modified=last_modified,
            checksum=checksum,
        )

    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
        raise


def build_dataset_metadata(
    dataset_path: Union[str, Path],
    pattern: str = "*.vla",
    compute_checksums: bool = False,
    force_rebuild: bool = False,
) -> MetadataManager:
    """
    Build or update metadata for an entire dataset.

    Args:
        dataset_path: Path to the dataset directory
        pattern: File pattern to match trajectory files
        compute_checksums: Whether to compute file checksums
        force_rebuild: Force rebuild even if metadata exists

    Returns:
        MetadataManager instance with loaded metadata
    """
    dataset_path = Path(dataset_path)
    manager = MetadataManager(dataset_path)

    # Check if metadata exists and we're not forcing rebuild
    if manager.exists() and not force_rebuild:
        logger.info(f"Metadata already exists at {manager.metadata_path}")
        return manager

    # Find all trajectory files
    if dataset_path.is_dir():
        trajectory_files = list(dataset_path.glob(pattern))
    else:
        # Single file case
        trajectory_files = [dataset_path]

    logger.info(f"Found {len(trajectory_files)} trajectory files")

    # Extract metadata for each file
    metadata_list = []
    for i, file_path in enumerate(trajectory_files):
        try:
            logger.debug(
                f"Processing {i+1}/{len(trajectory_files)}: {file_path}")
            metadata = extract_trajectory_metadata(str(file_path),
                                                   compute_checksums)
            metadata_list.append(metadata)
        except Exception as e:
            logger.warning(f"Skipping {file_path} due to error: {e}")
            continue

    # Save metadata
    if metadata_list:
        manager.save_metadata(metadata_list)
        logger.info(f"Built metadata for {len(metadata_list)} trajectories")
    else:
        logger.warning("No valid trajectories found")

    return manager


def update_dataset_metadata(
    dataset_path: Union[str, Path],
    pattern: str = "*.vla",
    compute_checksums: bool = False,
) -> MetadataManager:
    """
    Update metadata for new or modified files in the dataset.

    Args:
        dataset_path: Path to the dataset directory
        pattern: File pattern to match trajectory files
        compute_checksums: Whether to compute file checksums

    Returns:
        MetadataManager instance with updated metadata
    """
    dataset_path = Path(dataset_path)
    manager = MetadataManager(dataset_path)

    # Find all trajectory files
    if dataset_path.is_dir():
        trajectory_files = list(dataset_path.glob(pattern))
    else:
        trajectory_files = [dataset_path]

    # If no existing metadata, build from scratch
    if not manager.exists():
        return build_dataset_metadata(str(dataset_path), pattern,
                                      compute_checksums)

    # Load existing metadata
    existing_metadata = {
        meta.file_path: meta
        for meta in manager.get_all_metadata()
    }

    # Check for new or modified files
    updates_needed = []
    for file_path in trajectory_files:
        file_path_str = str(file_path.resolve())
        file_stat = os.stat(file_path_str)
        last_modified = datetime.fromtimestamp(file_stat.st_mtime)

        # Check if file is new or modified
        if (file_path_str not in existing_metadata
                or existing_metadata[file_path_str].last_modified
                < last_modified):
            try:
                metadata = extract_trajectory_metadata(file_path_str,
                                                       compute_checksums)
                updates_needed.append(metadata)
            except Exception as e:
                logger.warning(f"Skipping {file_path_str} due to error: {e}")

    # Update metadata if needed
    if updates_needed:
        manager.update_metadata(updates_needed)
        logger.info(f"Updated metadata for {len(updates_needed)} trajectories")
    else:
        logger.info("No metadata updates needed")

    return manager
