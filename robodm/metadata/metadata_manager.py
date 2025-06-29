import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import ray.data as rd

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetadata:
    """Metadata for a single trajectory."""

    file_path: str
    trajectory_length: int
    feature_keys: List[str]
    feature_shapes: Dict[str, List[int]]
    feature_dtypes: Dict[str, str]
    file_size: int
    last_modified: datetime
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime to string
        data["last_modified"] = self.last_modified.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryMetadata":
        """Create from dictionary."""
        # Convert string back to datetime
        data["last_modified"] = datetime.fromisoformat(data["last_modified"])
        return cls(**data)


class MetadataManager:
    """Manages trajectory metadata using Ray datasets for fast computation."""

    def __init__(
        self,
        ray_dataset: rd.Dataset,
    ):
        """
        Initialize metadata manager.

        Args:
            ray_dataset: Ray dataset instance for metadata computation
        """
        self.ray_dataset = ray_dataset
        self._metadata_cache: Optional[List[TrajectoryMetadata]] = None

    @classmethod
    def from_ray_dataset(
        cls,
        ray_dataset: rd.Dataset,
        auto_build: bool = True,
        **kwargs
    ) -> "MetadataManager":
        """
        Create MetadataManager from a Ray dataset.
        
        Args:
            ray_dataset: Ray dataset to manage metadata for
            auto_build: Whether to automatically build metadata if missing
            **kwargs: Additional arguments for MetadataManager
        """
        manager = cls(ray_dataset=ray_dataset, **kwargs)
        
        # Build metadata if requested
        if auto_build:
            manager.build_metadata()
            
        return manager

    def build_metadata(self, compute_checksums: bool = False) -> None:
        """
        Build metadata from the ray dataset.
        
        Args:
            compute_checksums: Whether to compute file checksums
        """
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

        # Use Ray Dataset for parallel processing
        metadata_dataset = self.ray_dataset.map(extract_metadata_ray)
        
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

        # Cache metadata
        self._metadata_cache = metadata_list
        logger.info(f"Built metadata for {len(metadata_list)} trajectories using Ray")

    def get_metadata(self, force_rebuild: bool = False) -> List[TrajectoryMetadata]:
        """
        Get metadata, building if necessary.

        Args:
            force_rebuild: Force rebuild metadata even if cached

        Returns:
            List of trajectory metadata
        """
        if self._metadata_cache is None or force_rebuild:
            self.build_metadata()
        
        return self._metadata_cache or []

    def get_trajectory_metadata(
            self, file_path: str) -> Optional[TrajectoryMetadata]:
        """
        Get metadata for a specific trajectory file.

        Args:
            file_path: Path to the trajectory file

        Returns:
            TrajectoryMetadata object or None if not found
        """
        metadata_list = self.get_metadata()

        # Normalize the file path for comparison
        file_path = str(Path(file_path).resolve())

        for metadata in metadata_list:
            if metadata.file_path == file_path:
                return metadata
        
        return None

    def get_all_metadata(self) -> List[TrajectoryMetadata]:
        """
        Get all trajectory metadata.

        Returns:
            List of TrajectoryMetadata objects
        """
        return self.get_metadata()

    def filter_by_length(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None) -> List[TrajectoryMetadata]:
        """
        Filter trajectories by length.

        Args:
            min_length: Minimum trajectory length
            max_length: Maximum trajectory length

        Returns:
            List of TrajectoryMetadata objects matching the criteria
        """
        metadata_list = self.get_metadata()
        
        filtered = []
        for metadata in metadata_list:
            if min_length is not None and metadata.trajectory_length < min_length:
                continue
            if max_length is not None and metadata.trajectory_length > max_length:
                continue
            filtered.append(metadata)
        
        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        metadata_list = self.get_metadata()
        
        if not metadata_list:
            return {
                "total_trajectories": 0,
                "total_timesteps": 0,
                "average_length": 0,
                "min_length": 0,
                "max_length": 0,
                "total_size_bytes": 0,
                "unique_feature_keys": [],
            }

        # Extract statistics
        lengths = [meta.trajectory_length for meta in metadata_list]
        sizes = [meta.file_size for meta in metadata_list]
        
        # Safely extract all unique feature keys
        all_feature_keys = []
        for metadata in metadata_list:
            if isinstance(metadata.feature_keys, list):
                all_feature_keys.extend(metadata.feature_keys)

        return {
            "total_trajectories": len(metadata_list),
            "total_timesteps": sum(lengths),
            "average_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_size_bytes": sum(sizes),
            "unique_feature_keys": list(set(all_feature_keys)),
        }
