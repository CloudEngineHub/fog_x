#!/usr/bin/env python3
"""
Validation Script for VLM Responses

This script validates VLM responses against ground truth data and calculates accuracy metrics.
It can work with various ground truth sources:
- Ground truth labels from filename patterns (success_*, failure_*)
- Ground truth labels from trajectory metadata
- Manual ground truth labels from JSON files

Usage:
    # Validate against filename patterns
    python validate_vlm_responses.py --results results.json --ground-truth-source filename

    # Validate against trajectory metadata
    python validate_vlm_responses.py --results results.json --ground-truth-source metadata --metadata-key "task_success"

    # Validate against manual labels
    python validate_vlm_responses.py --results results.json --ground-truth-source manual --ground-truth-file labels.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from robodm import Trajectory


def extract_ground_truth_from_filename(trajectory_path: str) -> Optional[bool]:
    """
    Extract ground truth label from filename pattern.
    
    Args:
        trajectory_path: Path to trajectory file
        
    Returns:
        True for success, False for failure, None if unclear
    """
    filename = os.path.basename(trajectory_path).lower()
    
    # Check for explicit success/failure patterns (handle underscores and other separators)
    if re.search(r'\bsuccess\b|success_', filename):
        return True
    elif re.search(r'\bfail(ure)?\b|fail(ure)?_', filename):
        return False
    
    # Check for directory-based patterns
    dir_path = os.path.dirname(trajectory_path).lower()
    if 'success' in dir_path:
        return True
    elif 'fail' in dir_path:
        return False
    
    return None


def extract_ground_truth_from_metadata(trajectory_path: str, metadata_key: str) -> Optional[bool]:
    """
    Extract ground truth label from trajectory metadata.
    
    Args:
        trajectory_path: Path to trajectory file
        metadata_key: Key in metadata containing ground truth
        
    Returns:
        True for success, False for failure, None if not found
    """
    try:
        traj = Trajectory(trajectory_path, mode="r")
        data = traj.load()
        traj.close()
        
        if metadata_key in data:
            value = data[metadata_key]
            
            # Handle various data types
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    value = value.item()
                else:
                    value = value[0]
            
            # Convert to boolean
            if isinstance(value, bool):
                return value
            elif isinstance(value, (int, float)):
                return bool(value)
            elif isinstance(value, str):
                value_lower = value.lower()
                if value_lower in {'true', 'success', 'successful', '1', 'yes'}:
                    return True
                elif value_lower in {'false', 'failure', 'failed', '0', 'no'}:
                    return False
        
        return None
        
    except Exception as e:
        print(f"⚠️  Error loading metadata from {trajectory_path}: {e}")
        return None


def load_manual_ground_truth(ground_truth_file: str) -> Dict[str, bool]:
    """
    Load manual ground truth labels from JSON file.
    
    Args:
        ground_truth_file: Path to JSON file with ground truth labels
        
    Returns:
        Dictionary mapping trajectory paths to ground truth labels
    """
    try:
        with open(ground_truth_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading ground truth file {ground_truth_file}: {e}")
        return {}


def extract_vlm_prediction(vlm_response: str, question: str) -> Optional[bool]:
    """
    Extract binary prediction from VLM response.
    
    Args:
        vlm_response: Raw VLM response text
        question: Original question asked
        
    Returns:
        True for positive, False for negative, None if unclear
    """
    if not vlm_response:
        return None
    
    response_lower = vlm_response.lower()
    
    # Look for clear Yes/No at the start of the response (most reliable)
    response_start = response_lower.strip()[:50]  # First 50 characters
    
    if re.match(r'^(yes|y)\b', response_start):
        return True
    elif re.match(r'^(no|n)\b', response_start):
        return False
    
    # Look for definitive statements in first sentence
    first_sentence = response_lower.split('.')[0] if '.' in response_lower else response_lower[:200]
    
    # Strong positive indicators in first sentence
    if re.search(r'\b(yes|successful|completed|achieved)\b', first_sentence):
        return True
    
    # Strong negative indicators in first sentence
    if re.search(r'\b(no|fail(ed|ure)?|unsuccessful|incomplete)\b', first_sentence):
        return False
    
    # Fallback: pattern matching with weights
    positive_patterns = [
        r'\byes\b', r'\btrue\b', r'\bsuccess(ful)?\b', r'\bcompleted?\b',
        r'\bachieved?\b', r'\baccomplished\b', r'\bworked?\b'
    ]
    
    negative_patterns = [
        r'\bno\b', r'\bfalse\b', r'\bfail(ed|ure)?\b', r'\bincomplete\b',
        r'\bunsuccessful\b', r'\bdid\s+not\b', r'\bdidn\'t\b'
    ]
    
    # Weight early occurrences more heavily
    first_100_chars = response_lower[:100]
    positive_early = sum(2 for pattern in positive_patterns if re.search(pattern, first_100_chars))
    negative_early = sum(2 for pattern in negative_patterns if re.search(pattern, first_100_chars))
    
    # Count all occurrences 
    positive_total = sum(1 for pattern in positive_patterns if re.search(pattern, response_lower))
    negative_total = sum(1 for pattern in negative_patterns if re.search(pattern, response_lower))
    
    total_positive = positive_early + positive_total
    total_negative = negative_early + negative_total
    
    if total_positive > total_negative and total_positive > 0:
        return True
    elif total_negative > total_positive and total_negative > 0:
        return False
    
    return None


def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: List of binary predictions
        ground_truth: List of binary ground truth labels
        
    Returns:
        Dictionary with accuracy, precision, recall, F1, and confusion matrix
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate confusion matrix components
    tp = np.sum((predictions == True) & (ground_truth == True))
    tn = np.sum((predictions == False) & (ground_truth == False))
    fp = np.sum((predictions == True) & (ground_truth == False))
    fn = np.sum((predictions == False) & (ground_truth == True))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn)
        }
    }


def validate_vlm_responses(
    results: Dict[str, Dict[str, Any]],
    ground_truth_source: str,
    metadata_key: Optional[str] = None,
    ground_truth_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate VLM responses against ground truth.
    
    Args:
        results: Results from VLM processing
        ground_truth_source: Source of ground truth ('filename', 'metadata', 'manual')
        metadata_key: Key for metadata-based ground truth
        ground_truth_file: File for manual ground truth
        
    Returns:
        Validation results with metrics and detailed comparisons
    """
    print(f"🔍 Validating {len(results)} VLM responses...")
    print(f"📊 Ground truth source: {ground_truth_source}")
    
    # Load manual ground truth if needed
    manual_gt = {}
    if ground_truth_source == "manual" and ground_truth_file:
        manual_gt = load_manual_ground_truth(ground_truth_file)
        print(f"📂 Loaded {len(manual_gt)} manual labels")
    
    # Process each result
    validated_results = []
    skipped_count = 0
    failed_processing_count = 0
    
    for trajectory_path, result in results.items():
        if not result["success"]:
            failed_processing_count += 1
            continue
        
        # Extract ground truth
        ground_truth = None
        if ground_truth_source == "filename":
            ground_truth = extract_ground_truth_from_filename(trajectory_path)
        elif ground_truth_source == "metadata" and metadata_key:
            ground_truth = extract_ground_truth_from_metadata(trajectory_path, metadata_key)
        elif ground_truth_source == "manual":
            # Try multiple key formats to handle path mismatches
            candidate_keys = [
                trajectory_path,  # Exact match
                os.path.basename(trajectory_path),  # Just filename
                os.path.splitext(os.path.basename(trajectory_path))[0],  # Filename without extension
                # Handle trajectory.h5 suffix removal
                trajectory_path.replace('/trajectory.h5', '') if trajectory_path.endswith('/trajectory.h5') else trajectory_path,
                # Handle directory path extraction for trajectory.h5 files
                os.path.dirname(trajectory_path) if trajectory_path.endswith('/trajectory.h5') else trajectory_path
            ]
            
            for key in candidate_keys:
                if key in manual_gt:
                    ground_truth = manual_gt[key]
                    break
        
        if ground_truth is None:
            skipped_count += 1
            continue
        
        # Extract VLM prediction - prefer pre-computed prediction from VLM results
        vlm_response = result.get("vlm_response", "")  # Always get VLM response for logging
        
        if "vlm_prediction" in result and result["vlm_prediction"] is not None:
            vlm_prediction = result["vlm_prediction"]
        else:
            # Fallback to parsing VLM response if no pre-computed prediction
            question = "question"  # We don't have access to original question here
            vlm_prediction = extract_vlm_prediction(vlm_response, question)
        
        if vlm_prediction is None:
            skipped_count += 1
            continue
        
        validated_results.append({
            "trajectory_path": trajectory_path,
            "ground_truth": ground_truth,
            "vlm_prediction": vlm_prediction,
            "vlm_response": vlm_response,
            "correct": ground_truth == vlm_prediction
        })
    
    print(f"✅ Validated: {len(validated_results)}")
    print(f"❌ Failed processing: {failed_processing_count}")
    print(f"⏩ Skipped (no ground truth): {skipped_count}")
    
    if len(validated_results) == 0:
        return {
            "error": "No valid comparisons found",
            "total_processed": len(results),
            "failed_processing": failed_processing_count,
            "skipped": skipped_count
        }
    
    # Calculate overall metrics
    predictions = [r["vlm_prediction"] for r in validated_results]
    ground_truths = [r["ground_truth"] for r in validated_results]
    metrics = calculate_metrics(predictions, ground_truths)
    
    return {
        "total_processed": len(results),
        "validated": len(validated_results),
        "failed_processing": failed_processing_count,
        "skipped": skipped_count,
        "metrics": metrics,
        "detailed_results": validated_results
    }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate VLM Responses Against Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate against filename patterns
    python validate_vlm_responses.py \\
        --results vlm_results.json \\
        --ground-truth-source filename

    # Validate against trajectory metadata
    python validate_vlm_responses.py \\
        --results vlm_results.json \\
        --ground-truth-source metadata \\
        --metadata-key "task_success"

    # Validate against manual labels
    python validate_vlm_responses.py \\
        --results vlm_results.json \\
        --ground-truth-source manual \\
        --ground-truth-file ground_truth.json
        """)
    
    parser.add_argument(
        "--results", 
        required=True,
        help="JSON file containing VLM processing results"
    )
    parser.add_argument(
        "--ground-truth-source", 
        choices=["filename", "metadata", "manual"],
        required=True,
        help="Source of ground truth labels"
    )
    parser.add_argument(
        "--metadata-key",
        help="Key in trajectory metadata for ground truth (required for metadata source)"
    )
    parser.add_argument(
        "--ground-truth-file",
        help="JSON file with manual ground truth labels (required for manual source)"
    )
    parser.add_argument(
        "--output",
        help="Output file for validation results (JSON format)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-trajectory results"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.ground_truth_source == "metadata" and not args.metadata_key:
        print("❌ --metadata-key is required when using metadata ground truth source")
        return 1
    
    if args.ground_truth_source == "manual" and not args.ground_truth_file:
        print("❌ --ground-truth-file is required when using manual ground truth source")
        return 1
    
    # Load VLM results
    try:
        with open(args.results, 'r') as f:
            results = json.load(f)
        print(f"📂 Loaded {len(results)} VLM results from {args.results}")
    except Exception as e:
        print(f"❌ Error loading results file {args.results}: {e}")
        return 1
    
    # Validate results
    try:
        validation_results = validate_vlm_responses(
            results=results,
            ground_truth_source=args.ground_truth_source,
            metadata_key=args.metadata_key,
            ground_truth_file=args.ground_truth_file
        )
        
        if "error" in validation_results:
            print(f"❌ Validation failed: {validation_results['error']}")
            return 1
        
        # Print summary
        metrics = validation_results["metrics"]
        cm = metrics["confusion_matrix"]
        
        print("\n📈 Validation Results")
        print("=" * 50)
        print(f"Total trajectories: {validation_results['total_processed']}")
        print(f"Successfully validated: {validation_results['validated']}")
        print(f"Failed processing: {validation_results['failed_processing']}")
        print(f"Skipped (no ground truth or prediction): {validation_results['skipped']}")
        
        print(f"\n🎯 Accuracy Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print("                 Predicted")
        print("                 Fail  Success")
        print(f"Actual   Fail    {cm['true_negative']:4d}  {cm['false_positive']:7d}")
        print(f"         Success {cm['false_negative']:4d}  {cm['true_positive']:7d}")
        
        # Show detailed results if requested
        if args.verbose:
            print(f"\n📝 Detailed Results:")
            print("-" * 60)
            for result in validation_results["detailed_results"]:
                status = "✅" if result["correct"] else "❌"
                filename = os.path.basename(result["trajectory_path"])
                print(f"{status} {filename}")
                print(f"   Ground Truth: {result['ground_truth']}")
                print(f"   VLM Prediction: {result['vlm_prediction']}")
                if not result["correct"]:
                    print(f"   VLM Response: {result['vlm_response'][:100]}...")
                print()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation_results, f, indent=2)
            print(f"💾 Validation results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())