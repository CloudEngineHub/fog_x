import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import robodm
from robodm.metadata_manager import MetadataManager, TrajectoryMetadata
from robodm.dataset import VLADataset

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
    Build or update metadata for an entire dataset using Ray for fast parallel processing.

    Args:
        dataset_path: Path to the dataset directory
        pattern: File pattern to match trajectory files
        compute_checksums: Whether to compute file checksums
        force_rebuild: Force rebuild even if metadata exists

    Returns:
        MetadataManager instance with loaded metadata
    """
    dataset_path = Path(dataset_path)
    
    # Create VLADataset for Ray-based processing
    dataset = VLADataset.create_trajectory_dataset(
        path=str(dataset_path / pattern),
        return_type="numpy"
    )
    
    manager = MetadataManager(dataset)

    # Check if metadata exists and we're not forcing rebuild
    if manager.exists() and not force_rebuild:
        logger.info(f"Metadata already exists at {manager.metadata_path}")
        return manager

    def extract_metadata_ray(row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a single trajectory using Ray."""
        import hashlib
        from datetime import datetime
        
        # Get file path from row metadata
        file_path = row.get('__file_path__', 'unknown')
        
        # Extract trajectory length from first data key
        data_keys = [k for k in row.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError("No data keys found in row")
        
        first_key = data_keys[0]
        first_value = row[first_key]
        
        if hasattr(first_value, '__len__'):
            trajectory_length = len(first_value)
        else:
            trajectory_length = 1
        
        # Extract feature information
        feature_keys = data_keys
        feature_shapes = {}
        feature_dtypes = {}
        
        for key in feature_keys:
            value = row[key]
            if hasattr(value, "shape"):
                # For numpy arrays - exclude time dimension
                shape = list(value.shape)
                feature_shapes[key] = shape[1:] if len(shape) > 1 else []
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
        
        # Get file metadata if path exists
        if file_path != 'unknown' and os.path.exists(file_path):
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Compute checksum if requested
            checksum = None
            if compute_checksums:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
                checksum = sha256_hash.hexdigest()
        else:
            file_size = 0
            last_modified = datetime.now()
            checksum = None
        
        return {
            'file_path': file_path,
            'trajectory_length': trajectory_length,
            'feature_keys': feature_keys,
            'feature_shapes': feature_shapes,
            'feature_dtypes': feature_dtypes,
            'file_size': file_size,
            'last_modified': last_modified.isoformat(),
            'checksum': checksum
        }

    # Use Ray Dataset for parallel processing instead of for loop
    ray_dataset = dataset.get_ray_dataset()
    metadata_dataset = ray_dataset.map(extract_metadata_ray)
    
    # Collect results and convert to TrajectoryMetadata objects
    metadata_list = []
    for metadata_dict in metadata_dataset.take_all():
        # Convert datetime string back to datetime object
        metadata_dict['last_modified'] = datetime.fromisoformat(metadata_dict['last_modified'])
        
        metadata = TrajectoryMetadata(
            file_path=metadata_dict['file_path'],
            trajectory_length=metadata_dict['trajectory_length'],
            feature_keys=metadata_dict['feature_keys'],
            feature_shapes=metadata_dict['feature_shapes'],
            feature_dtypes=metadata_dict['feature_dtypes'],
            file_size=metadata_dict['file_size'],
            last_modified=metadata_dict['last_modified'],
            checksum=metadata_dict['checksum']
        )
        metadata_list.append(metadata)

    # Save metadata
    if metadata_list:
        manager.save_metadata(metadata_list)
        logger.info(f"Built metadata for {len(metadata_list)} trajectories using Ray")
    else:
        logger.warning("No valid trajectories found")

    return manager


def update_dataset_metadata(
    dataset_path: Union[str, Path],
    pattern: str = "*.vla",
    compute_checksums: bool = False,
) -> MetadataManager:
    """
    Update metadata for new or modified files in the dataset using Ray.

    Args:
        dataset_path: Path to the dataset directory
        pattern: File pattern to match trajectory files
        compute_checksums: Whether to compute file checksums

    Returns:
        MetadataManager instance with updated metadata
    """
    dataset_path = Path(dataset_path)
    
    # Create VLADataset for Ray-based processing
    dataset = VLADataset.create_trajectory_dataset(
        path=str(dataset_path / pattern),
        return_type="numpy"
    )
    
    manager = MetadataManager(dataset)

    # If no existing metadata, build from scratch
    if not manager.exists():
        return build_dataset_metadata(str(dataset_path), pattern, compute_checksums)

    # Load existing metadata
    existing_metadata = {
        meta.file_path: meta
        for meta in manager.get_all_metadata()
    }

    # Find all trajectory files
    if dataset_path.is_dir():
        trajectory_files = list(dataset_path.glob(pattern))
    else:
        trajectory_files = [dataset_path]

    # Check for new or modified files
    files_to_update = []
    for file_path in trajectory_files:
        file_path_str = str(file_path.resolve())
        file_stat = os.stat(file_path_str)
        last_modified = datetime.fromtimestamp(file_stat.st_mtime)

        # Check if file is new or modified
        if (file_path_str not in existing_metadata
                or existing_metadata[file_path_str].last_modified < last_modified):
            files_to_update.append(file_path_str)

    if not files_to_update:
        logger.info("No metadata updates needed")
        return manager

    # Filter dataset to only include files that need updating
    def filter_updated_files(row: Dict[str, Any]) -> bool:
        file_path = row.get('__file_path__', 'unknown')
        return file_path in files_to_update

    # Use Ray to process only the files that need updating
    ray_dataset = dataset.get_ray_dataset()
    filtered_dataset = ray_dataset.filter(filter_updated_files)
    
    # Same metadata extraction function as in build_dataset_metadata
    def extract_metadata_ray(row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a single trajectory using Ray."""
        import hashlib
        from datetime import datetime
        
        # Get file path from row metadata
        file_path = row.get('__file_path__', 'unknown')
        
        # Extract trajectory length from first data key
        data_keys = [k for k in row.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError("No data keys found in row")
        
        first_key = data_keys[0]
        first_value = row[first_key]
        
        if hasattr(first_value, '__len__'):
            trajectory_length = len(first_value)
        else:
            trajectory_length = 1
        
        # Extract feature information
        feature_keys = data_keys
        feature_shapes = {}
        feature_dtypes = {}
        
        for key in feature_keys:
            value = row[key]
            if hasattr(value, "shape"):
                # For numpy arrays - exclude time dimension
                shape = list(value.shape)
                feature_shapes[key] = shape[1:] if len(shape) > 1 else []
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
        
        # Get file metadata if path exists
        if file_path != 'unknown' and os.path.exists(file_path):
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Compute checksum if requested
            checksum = None
            if compute_checksums:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
                checksum = sha256_hash.hexdigest()
        else:
            file_size = 0
            last_modified = datetime.now()
            checksum = None
        
        return {
            'file_path': file_path,
            'trajectory_length': trajectory_length,
            'feature_keys': feature_keys,
            'feature_shapes': feature_shapes,
            'feature_dtypes': feature_dtypes,
            'file_size': file_size,
            'last_modified': last_modified.isoformat(),
            'checksum': checksum
        }

    metadata_dataset = filtered_dataset.map(extract_metadata_ray)
    
    # Collect results and convert to TrajectoryMetadata objects
    updates_needed = []
    for metadata_dict in metadata_dataset.take_all():
        # Convert datetime string back to datetime object
        metadata_dict['last_modified'] = datetime.fromisoformat(metadata_dict['last_modified'])
        
        metadata = TrajectoryMetadata(
            file_path=metadata_dict['file_path'],
            trajectory_length=metadata_dict['trajectory_length'],
            feature_keys=metadata_dict['feature_keys'],
            feature_shapes=metadata_dict['feature_shapes'],
            feature_dtypes=metadata_dict['feature_dtypes'],
            file_size=metadata_dict['file_size'],
            last_modified=metadata_dict['last_modified'],
            checksum=metadata_dict['checksum']
        )
        updates_needed.append(metadata)

    # Update metadata if needed
    if updates_needed:
        manager.update_metadata(updates_needed)
        logger.info(f"Updated metadata for {len(updates_needed)} trajectories using Ray")
    else:
        logger.info("No metadata updates needed")

    return manager
