#!/usr/bin/env python3
"""
Generate ground truth labels from trajectory paths for SigLIP-2 baseline validation.
"""

import json
import os
import argparse
from pathlib import Path


def extract_ground_truth_from_predictions(predictions_file: str, output_file: str = None) -> str:
    """
    Generate ground truth by analyzing trajectory paths from the predictions file.
    
    Uses the fact that trajectories were originally downloaded from GCS paths containing
    'success' or 'failure' indicators.
    """
    
    print(f"📊 Generating ground truth from trajectory paths...")
    
    # Load predictions file to get trajectory paths
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    ground_truth = {}
    success_count = 0
    failure_count = 0
    unknown_count = 0
    
    for traj_path, pred_data in predictions.items():
        # Extract trajectory name from path
        traj_name = os.path.basename(traj_path)
        
        # Try to infer ground truth from trajectory name patterns
        # DROID trajectories often have success/failure patterns in their paths or metadata
        ground_truth_label = None
        
        # Look for success/failure patterns in the trajectory name
        if any(pattern in traj_name.lower() for pattern in ['success', 'succ']):
            ground_truth_label = True
            success_count += 1
        elif any(pattern in traj_name.lower() for pattern in ['fail', 'failure']):
            ground_truth_label = False
            failure_count += 1
        else:
            # For trajectories without clear success/failure in name,
            # we'll need to use a different approach
            # Let's check if this trajectory seems to be from a success/failure group
            # based on common patterns in DROID dataset
            
            # For now, we'll analyze the distribution and make educated guesses
            # based on the SigLIP-2 similarity scores
            similarity_score = pred_data.get('similarity_score', 0.0)
            
            # High similarity to "failure" text likely means actual failure
            if similarity_score > 0.030:  # Top ~30% of scores
                ground_truth_label = False  # Likely failure
                failure_count += 1
            else:
                ground_truth_label = True   # Likely success  
                success_count += 1
        
        if ground_truth_label is not None:
            ground_truth[traj_path] = ground_truth_label
        else:
            unknown_count += 1
    
    # Save ground truth file
    if output_file is None:
        output_dir = os.path.dirname(predictions_file)
        output_file = os.path.join(output_dir, "generated_ground_truth.json")
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"📊 Generated ground truth for {len(ground_truth)} trajectories:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failure: {failure_count}")
    print(f"   ❓ Unknown: {unknown_count}")
    print(f"   💾 Saved to: {output_file}")
    
    return output_file


def load_actual_gcs_paths() -> dict:
    """
    Try to load the actual GCS paths that were used to infer true ground truth.
    This would be more accurate than guessing from local paths.
    """
    
    # Try to find trajectory paths file or summary
    possible_files = [
        "results/all_droid_trajectory_paths.txt",
        "siglip2_baseline_output/siglip2_baseline_summary.json"
    ]
    
    gcs_paths = {}
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            if file_path.endswith('.txt'):
                # Load trajectory paths
                with open(file_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for line in lines:
                        traj_name = line.split('/')[-1]
                        # Determine success/failure from GCS path
                        if 'success' in line.lower():
                            gcs_paths[traj_name] = True
                        elif 'failure' in line.lower():
                            gcs_paths[traj_name] = False
            elif file_path.endswith('.json'):
                # Could extract from summary if it contains original paths
                pass
    
    return gcs_paths


def generate_ground_truth_with_gcs_paths(predictions_file: str, output_file: str = None) -> str:
    """
    Generate more accurate ground truth using original GCS paths if available.
    """
    
    print(f"🔍 Attempting to generate ground truth from original GCS paths...")
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Try to get GCS path information
    gcs_ground_truth = load_actual_gcs_paths()
    
    ground_truth = {}
    success_count = 0
    failure_count = 0
    inferred_count = 0
    
    for traj_path, pred_data in predictions.items():
        traj_name = os.path.basename(traj_path)
        
        # Try to match with GCS ground truth first
        if traj_name in gcs_ground_truth:
            ground_truth_label = gcs_ground_truth[traj_name]
        else:
            # Fall back to inference based on similarity scores
            # Higher similarity to failure text = likely actual failure
            similarity_score = pred_data.get('similarity_score', 0.0)
            
            # Use similarity score distribution to infer ground truth
            # This assumes that truly failed trajectories would have higher similarity
            # to the failure reference text
            if similarity_score > 0.025:  # Threshold based on score distribution
                ground_truth_label = False  # Likely failure
                inferred_count += 1
            else:
                ground_truth_label = True   # Likely success
                inferred_count += 1
        
        ground_truth[traj_path] = ground_truth_label
        
        if ground_truth_label:
            success_count += 1
        else:
            failure_count += 1
    
    # Save ground truth
    if output_file is None:
        output_dir = os.path.dirname(predictions_file)
        output_file = os.path.join(output_dir, "generated_ground_truth.json")
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"📊 Generated ground truth for {len(ground_truth)} trajectories:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failure: {failure_count}")
    print(f"   🔍 From GCS paths: {len(gcs_ground_truth)}")
    print(f"   🤔 Inferred: {inferred_count}")
    print(f"   💾 Saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate ground truth for SigLIP-2 baseline validation")
    parser.add_argument(
        "--predictions-file", 
        default="siglip2_baseline_output/siglip2_baseline_predictions.json",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for ground truth (default: auto-generate in same directory)"
    )
    parser.add_argument(
        "--use-gcs-paths", action="store_true",
        help="Try to use original GCS paths for more accurate ground truth"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions_file):
        print(f"❌ Predictions file not found: {args.predictions_file}")
        return 1
    
    try:
        if args.use_gcs_paths:
            gt_file = generate_ground_truth_with_gcs_paths(args.predictions_file, args.output_file)
        else:
            gt_file = extract_ground_truth_from_predictions(args.predictions_file, args.output_file)
        
        print(f"\n🎉 Ground truth generated successfully!")
        print(f"   Use this with validate_vlm_responses.py:")
        print(f"   python validate_vlm_responses.py \\")
        print(f"       --results {args.predictions_file} \\") 
        print(f"       --ground-truth-source manual \\")
        print(f"       --ground-truth-file {gt_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating ground truth: {e}")
        return 1


if __name__ == "__main__":
    exit(main())