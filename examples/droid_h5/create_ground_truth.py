#!/usr/bin/env python3
"""
Create ground truth file from DROID metadata for validation.
"""

import json
import os
import glob
from pathlib import Path

def create_ground_truth_from_metadata(results_dir, output_file):
    """
    Create a manual ground truth file from DROID metadata files.
    
    Args:
        results_dir: Directory containing DROID trajectories  
        output_file: Output JSON file for ground truth labels
    """
    ground_truth = {}
    
    # Find all metadata files
    metadata_files = glob.glob(os.path.join(results_dir, "droid_trajectories", "*", "metadata_*.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract trajectory directory name
            trajectory_dir = os.path.dirname(metadata_file)
            trajectory_name = os.path.basename(trajectory_dir)
            
            # Create the path format used in VLM results
            trajectory_path = f"./results/droid_trajectories/{trajectory_name}"
            
            # Extract success label
            success = metadata.get("success", None)
            if success is not None:
                ground_truth[trajectory_path] = success
                print(f"Added: {trajectory_name} -> {success}")
                
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
            continue
    
    # Save ground truth file
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\nCreated ground truth file: {output_file}")
    print(f"Total trajectories: {len(ground_truth)}")
    
    # Count success/failure
    successful = sum(1 for v in ground_truth.values() if v)
    failed = sum(1 for v in ground_truth.values() if not v)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return ground_truth

if __name__ == "__main__":
    results_dir = "./results"
    output_file = "./results/ground_truth.json"
    
    create_ground_truth_from_metadata(results_dir, output_file)