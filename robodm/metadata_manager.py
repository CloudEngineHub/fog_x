import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, asdict
from datetime import datetime

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
        data['last_modified'] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryMetadata':
        """Create from dictionary."""
        # Convert string back to datetime
        data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


class MetadataManager:
    """Manages parquet metadata files for trajectory datasets."""
    
    def __init__(self, dataset_path: Union[str, Path], metadata_filename: str = "trajectory_metadata.parquet"):
        """
        Initialize metadata manager.
        
        Args:
            dataset_path: Path to the dataset directory
            metadata_filename: Name of the metadata parquet file
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / metadata_filename
        self._metadata_cache: Optional[pd.DataFrame] = None
        
    def exists(self) -> bool:
        """Check if metadata file exists."""
        return self.metadata_path.exists()
    
    def load_metadata(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load metadata from parquet file.
        
        Args:
            force_reload: Force reload from disk even if cached
            
        Returns:
            DataFrame with trajectory metadata
        """
        if self._metadata_cache is not None and not force_reload:
            return self._metadata_cache
            
        if not self.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        try:
            self._metadata_cache = pd.read_parquet(self.metadata_path)
            logger.info(f"Loaded metadata for {len(self._metadata_cache)} trajectories")
            return self._metadata_cache
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def save_metadata(self, metadata_list: List[TrajectoryMetadata]) -> None:
        """
        Save metadata to parquet file.
        
        Args:
            metadata_list: List of trajectory metadata objects
        """
        if not metadata_list:
            logger.warning("No metadata to save")
            return
            
        # Convert to DataFrame
        data = [meta.to_dict() for meta in metadata_list]
        df = pd.DataFrame(data)
        
        # Save to parquet
        try:
            df.to_parquet(self.metadata_path, index=False)
            self._metadata_cache = df
            logger.info(f"Saved metadata for {len(df)} trajectories to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def get_trajectory_metadata(self, file_path: str) -> Optional[TrajectoryMetadata]:
        """
        Get metadata for a specific trajectory file.
        
        Args:
            file_path: Path to the trajectory file
            
        Returns:
            TrajectoryMetadata object or None if not found
        """
        df = self.load_metadata()
        
        # Normalize the file path for comparison
        file_path = str(Path(file_path).resolve())
        
        matching_rows = df[df['file_path'] == file_path]
        if matching_rows.empty:
            return None
            
        # Convert back to TrajectoryMetadata object
        row = matching_rows.iloc[0].to_dict()
        return TrajectoryMetadata.from_dict(row)
    
    def update_metadata(self, new_metadata: List[TrajectoryMetadata]) -> None:
        """
        Update metadata for specific trajectories.
        
        Args:
            new_metadata: List of updated trajectory metadata
        """
        if not self.exists():
            # If no existing metadata, just save the new ones
            self.save_metadata(new_metadata)
            return
            
        df = self.load_metadata()
        
        # Create a mapping of file paths to new metadata
        update_map = {meta.file_path: meta.to_dict() for meta in new_metadata}
        
        # Update existing rows
        for idx, row in df.iterrows():
            if row['file_path'] in update_map:
                for key, value in update_map[row['file_path']].items():
                    df.at[idx, key] = value
                del update_map[row['file_path']]
        
        # Add new rows for files not in existing metadata
        if update_map:
            new_df = pd.DataFrame(list(update_map.values()))
            df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated metadata
        df.to_parquet(self.metadata_path, index=False)
        self._metadata_cache = df
        logger.info(f"Updated metadata for {len(new_metadata)} trajectories")
    
    def remove_metadata(self, file_paths: List[str]) -> None:
        """
        Remove metadata for specific trajectory files.
        
        Args:
            file_paths: List of file paths to remove
        """
        if not self.exists():
            logger.warning("No metadata file to remove from")
            return
            
        df = self.load_metadata()
        
        # Normalize file paths
        file_paths = [str(Path(fp).resolve()) for fp in file_paths]
        
        # Remove matching rows
        df = df[~df['file_path'].isin(file_paths)]
        
        # Save updated metadata
        df.to_parquet(self.metadata_path, index=False)
        self._metadata_cache = df
        logger.info(f"Removed metadata for {len(file_paths)} trajectories")
    
    def get_all_metadata(self) -> List[TrajectoryMetadata]:
        """
        Get all trajectory metadata.
        
        Returns:
            List of TrajectoryMetadata objects
        """
        df = self.load_metadata()
        return [TrajectoryMetadata.from_dict(row.to_dict()) for _, row in df.iterrows()]
    
    def filter_by_length(self, min_length: Optional[int] = None, max_length: Optional[int] = None) -> List[TrajectoryMetadata]:
        """
        Filter trajectories by length.
        
        Args:
            min_length: Minimum trajectory length
            max_length: Maximum trajectory length
            
        Returns:
            List of TrajectoryMetadata objects matching the criteria
        """
        df = self.load_metadata()
        
        if min_length is not None:
            df = df[df['trajectory_length'] >= min_length]
        if max_length is not None:
            df = df[df['trajectory_length'] <= max_length]
            
        return [TrajectoryMetadata.from_dict(row.to_dict()) for _, row in df.iterrows()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        df = self.load_metadata()
        
        # Safely extract all unique feature keys
        all_feature_keys = []
        for keys in df['feature_keys'].tolist():
            if isinstance(keys, list):
                all_feature_keys.extend(keys)
        
        return {
            'total_trajectories': len(df),
            'total_timesteps': df['trajectory_length'].sum(),
            'average_length': df['trajectory_length'].mean(),
            'min_length': df['trajectory_length'].min(),
            'max_length': df['trajectory_length'].max(),
            'total_size_bytes': df['file_size'].sum(),
            'unique_feature_keys': list(set(all_feature_keys))
        }