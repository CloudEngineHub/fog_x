#!/usr/bin/env python3
"""
Complete DROID HDF5 Pipeline: Download → Convert → Process → Validate

This script provides a complete end-to-end workflow similar to droid_to_robodm.py
but using the new HDF5 backend and VLM processing pipeline.

Features:
- Download DROID trajectories from GCS with gsutil
- Convert to HDF5 format for efficient processing
- Process trajectories with VLM for success/failure classification
- Validate results and generate comprehensive metrics
- Parallel processing with Ray for scalability
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

import ray
import numpy as np

# Add RoboDM to path
import sys
sys.path.append('/home/syx/ucsf/robodm')
import robodm
from robodm import Trajectory

# Import our pipeline components
from simple_vlm_processing import process_trajectories_parallel
from validate_vlm_responses import validate_vlm_responses


@ray.remote(num_cpus=1)
def download_and_convert_trajectory(
    trajectory_gcs_path: str, 
    output_dir: str, 
    temp_dir: str
) -> Tuple[bool, str, str, str]:
    """
    Download DROID trajectory from GCS and convert to HDF5.
    
    Args:
        trajectory_gcs_path: GCS path to DROID trajectory
        output_dir: Directory to save HDF5 trajectories
        temp_dir: Temporary directory for downloads
        
    Returns:
        Tuple of (success: bool, h5_path: str, error_msg: str, trajectory_name: str)
    """
    try:
        # Extract trajectory name from GCS path
        traj_name = trajectory_gcs_path.rstrip("/").split("/")[-1]
        
        # Determine success/failure from path
        success_label = "success" if "success" in trajectory_gcs_path else "failure"
        
        # Create local download path
        local_download_dir = os.path.join(temp_dir, traj_name)
        os.makedirs(os.path.dirname(local_download_dir), exist_ok=True)
        
        print(f"  📥 Downloading {traj_name}")
        
        # Download using gsutil
        result = subprocess.run([
            "gsutil", "-m", "cp", "-r", trajectory_gcs_path, temp_dir
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return False, "", f"gsutil download failed: {result.stderr}", traj_name
        
        # Convert to HDF5 using DROID processor
        sys.path.append('/home/syx/ucsf/robodm/examples/droid')
        from droid_to_robodm import DROIDProcessor
        
        processor = DROIDProcessor()
        
        print(f"  🔄 Converting {traj_name} to HDF5")
        
        # Load DROID data
        droid_data = processor.load_droid_trajectory(local_download_dir)
        
        # Generate HDF5 output path
        h5_filename = f"{success_label}_{traj_name}.h5"
        h5_path = os.path.join(output_dir, h5_filename)
        
        # Convert to RoboDM HDF5 format (backend determined by .h5 extension)
        processor.convert_to_robodm(droid_data, h5_path)
        
        # Clean up downloaded files
        if os.path.exists(local_download_dir):
            shutil.rmtree(local_download_dir)
        
        print(f"  ✅ Converted: {h5_filename}")
        return True, h5_path, "", traj_name
        
    except subprocess.TimeoutExpired:
        return False, "", f"Download timeout for {traj_name}", traj_name
    except Exception as e:
        import traceback
        error_msg = f"Error processing {traj_name}: {e}\n{traceback.format_exc()}"
        return False, "", error_msg, traj_name


def download_and_convert_trajectories(
    trajectory_paths: List[str],
    output_dir: str,
    max_workers: int = 4
) -> Tuple[List[str], List[str]]:
    """
    Download and convert multiple DROID trajectories to HDF5.
    
    Args:
        trajectory_paths: List of GCS paths to DROID trajectories
        output_dir: Directory to save HDF5 trajectories
        max_workers: Maximum parallel workers
        
    Returns:
        Tuple of (successful_h5_paths, failed_trajectories)
    """
    print(f"🚀 Starting download and conversion of {len(trajectory_paths)} trajectories")
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    # Create output and temp directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="droid_download_")
    
    try:
        # Submit all download/conversion tasks
        futures = []
        for traj_path in trajectory_paths:
            future = download_and_convert_trajectory.remote(
                traj_path, output_dir, temp_dir
            )
            futures.append(future)
        
        # Collect results
        successful_paths = []
        failed_trajectories = []
        completed = 0
        start_time = time.time()
        
        while futures:
            # Wait for at least one task to complete
            ready, futures = ray.wait(futures, num_returns=1, timeout=60.0)
            
            for future in ready:
                success, h5_path, error_msg, traj_name = ray.get(future)
                completed += 1
                
                if success:
                    successful_paths.append(h5_path)
                    status = "✅"
                else:
                    failed_trajectories.append(traj_name)
                    print(f"    ❌ {error_msg}")
                    status = "❌"
                
                # Progress update
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(trajectory_paths) - completed) / rate if rate > 0 else 0
                
                print(f"{status} [{completed}/{len(trajectory_paths)}] {traj_name} "
                      f"(Rate: {rate:.1f}/min, ETA: {eta/60:.1f}min)")
        
        total_time = time.time() - start_time
        print(f"\n📊 Download & Conversion Summary:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Successful: {len(successful_paths)}")
        print(f"  Failed: {len(failed_trajectories)}")
        print(f"  Rate: {len(trajectory_paths)/total_time*60:.1f} trajectories/minute")
        
        return successful_paths, failed_trajectories
        
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def run_complete_pipeline(
    trajectory_gcs_paths: List[str],
    output_dir: str,
    image_key: str = "observation/images/exterior_image_1_left",
    language_key: str = "metadata/language_instruction",
    question: str = "Is this trajectory successful?",
    max_workers: int = 4,
    skip_download: bool = False
) -> Dict:
    """
    Run complete pipeline: download → convert → process → validate.
    
    Args:
        trajectory_gcs_paths: GCS paths to DROID trajectories
        output_dir: Output directory for all files
        image_key: Key to extract images from trajectories
        language_key: Key to extract language instructions
        question: Question for VLM analysis
        max_workers: Maximum parallel workers
        skip_download: Skip download/conversion if HDF5 files already exist
        
    Returns:
        Dictionary with comprehensive pipeline results
    """
    print("🎯 DROID HDF5 Pipeline - Complete End-to-End Workflow")
    print("=" * 70)
    
    pipeline_start = time.time()
    h5_dir = os.path.join(output_dir, "hdf5_trajectories")
    results = {
        "input_trajectories": len(trajectory_gcs_paths),
        "stages": {}
    }
    
    # Stage 1: Download and Convert
    if skip_download:
        print("⏩ Skipping download/conversion - using existing HDF5 files")
        h5_files = list(Path(h5_dir).glob("*.h5"))
        successful_paths = [str(f) for f in h5_files]
        failed_downloads = []
    else:
        print("\n📥 Stage 1: Download and Convert DROID → HDF5")
        print("-" * 50)
        successful_paths, failed_downloads = download_and_convert_trajectories(
            trajectory_gcs_paths, h5_dir, max_workers
        )
    
    results["stages"]["download_convert"] = {
        "successful": len(successful_paths),
        "failed": len(failed_downloads) if not skip_download else 0,
        "h5_files": successful_paths
    }
    
    if not successful_paths:
        print("❌ No trajectories were successfully converted!")
        return results
    
    # Stage 2: VLM Processing
    print("\n🤖 Stage 2: VLM Processing")
    print("-" * 30)
    
    vlm_results_file = os.path.join(output_dir, "vlm_results.json")
    
    vlm_results = process_trajectories_parallel(
        trajectory_paths=successful_paths,
        image_key=image_key,
        language_key=language_key,
        question=question,
        max_workers=max_workers
    )
    
    # Save VLM results
    with open(vlm_results_file, 'w') as f:
        json.dump(vlm_results, f, indent=2)
    
    vlm_successful = sum(1 for r in vlm_results.values() if r["success"])
    vlm_failed = len(vlm_results) - vlm_successful
    
    results["stages"]["vlm_processing"] = {
        "total_processed": len(vlm_results),
        "successful": vlm_successful,
        "failed": vlm_failed,
        "results_file": vlm_results_file
    }
    
    print(f"📊 VLM Processing: {vlm_successful} successful, {vlm_failed} failed")
    
    # Stage 3: Validation
    print("\n✅ Stage 3: Validation")
    print("-" * 25)
    
    validation_results = validate_vlm_responses(
        results=vlm_results,
        ground_truth_source="filename"
    )
    
    validation_file = os.path.join(output_dir, "validation_results.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    if "error" not in validation_results:
        metrics = validation_results["metrics"]
        cm = metrics["confusion_matrix"]
        
        results["stages"]["validation"] = {
            "validated": validation_results["validated"],
            "skipped": validation_results["skipped"],
            "metrics": metrics,
            "results_file": validation_file
        }
        
        print(f"📈 Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print("                 Predicted")
        print("                 Fail  Success")
        print(f"Actual   Fail    {cm['true_negative']:4d}  {cm['false_positive']:7d}")
        print(f"         Success {cm['false_negative']:4d}  {cm['true_positive']:7d}")
    else:
        print(f"❌ Validation failed: {validation_results['error']}")
        results["stages"]["validation"] = {"error": validation_results["error"]}
    
    # Pipeline Summary
    total_time = time.time() - pipeline_start
    results["total_time"] = total_time
    
    print(f"\n🎉 Pipeline Complete!")
    print(f"📊 Total time: {total_time/60:.1f} minutes")
    print(f"📁 All results saved to: {output_dir}")
    
    # Save pipeline summary
    summary_file = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Complete DROID HDF5 Pipeline: Download → Convert → Process → Validate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline on success/failure trajectories
    python droid_hdf5_pipeline.py \\
        --trajectories gs://gresearch/robotics/droid_raw/1.0.1/success/episode_1 \\
                      gs://gresearch/robotics/droid_raw/1.0.1/failure/episode_2 \\
        --output-dir ./droid_hdf5_results \\
        --question "Is this trajectory successful?"

    # Use existing HDF5 files (skip download)
    python droid_hdf5_pipeline.py \\
        --trajectories dummy_path \\  # Not used when --skip-download
        --output-dir ./existing_results \\
        --skip-download \\
        --question "Did the robot complete the task successfully?"

    # Custom image and language keys
    python droid_hdf5_pipeline.py \\
        --trajectories gs://path/to/trajectories/*.tar \\
        --output-dir ./results \\
        --image-key "observation/images/wrist_camera" \\
        --language-key "metadata/task_description" \\
        --question "What task is the robot performing?"
        """)
    
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="GCS paths to DROID trajectory directories"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all pipeline results"
    )
    parser.add_argument(
        "--image-key",
        default="observation/images/exterior_image_1_left",
        help="Key to extract images from trajectories (default: exterior_image_1_left)"
    )
    parser.add_argument(
        "--language-key",
        default="metadata/language_instruction",
        help="Key to extract language instructions (default: metadata/language_instruction)"
    )
    parser.add_argument(
        "--question",
        default="Is this trajectory successful?",
        help="Question for VLM analysis"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers for processing"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download/conversion and use existing HDF5 files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running"
    )
    
    args = parser.parse_args()
    
    # Validate gsutil availability if not skipping download
    if not args.skip_download and not args.dry_run:
        try:
            subprocess.run(["gsutil", "version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ gsutil not found! Please install Google Cloud SDK:")
            print("   https://cloud.google.com/sdk/docs/install")
            return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dry_run:
        print("🔍 Dry Run - Pipeline Configuration")
        print("=" * 50)
        print(f"Input trajectories: {len(args.trajectories)}")
        for i, path in enumerate(args.trajectories, 1):
            print(f"  {i}. {path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Image key: {args.image_key}")
        print(f"Language key: {args.language_key}")
        print(f"VLM question: {args.question}")
        print(f"Max workers: {args.max_workers}")
        print(f"Skip download: {args.skip_download}")
        return 0
    
    try:
        results = run_complete_pipeline(
            trajectory_gcs_paths=args.trajectories,
            output_dir=args.output_dir,
            image_key=args.image_key,
            language_key=args.language_key,
            question=args.question,
            max_workers=args.max_workers,
            skip_download=args.skip_download
        )
        
        # Check if pipeline was successful
        validation_stage = results["stages"].get("validation", {})
        if "metrics" in validation_stage:
            accuracy = validation_stage["metrics"]["accuracy"]
            if accuracy >= 0.8:
                print(f"\n🎉 Pipeline completed successfully with {accuracy:.1%} accuracy!")
                return 0
            else:
                print(f"\n⚠️  Pipeline completed with low accuracy: {accuracy:.1%}")
                return 0
        else:
            print(f"\n❌ Pipeline completed with validation errors")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    exit(main())