#!/usr/bin/env python3
"""
Comprehensive GCS trajectory scanner for DROID dataset.

This script scans the entire DROID GCS bucket and creates a comprehensive
list of all available trajectory paths. This file can then be used by
droid_pipeline.py to randomly sample trajectories without re-scanning.
"""

import subprocess
import time
import argparse
from typing import List, Set
import ray
from functools import partial


@ray.remote
def scan_lab_trajectories(lab: str, base_path: str) -> List[str]:
    """
    Scan trajectories for a single lab in parallel.
    
    Args:
        lab: Lab name to scan
        base_path: Base GCS path
        
    Returns:
        List of trajectory paths for this lab
    """
    print(f"🔎 Scanning {lab}...")
    
    lab_trajectories = []
    
    for category in ['success', 'failure']:
        search_path = f"{base_path}{lab}/{category}/"
        print(f"  📂 {lab}/{category}...")
        
        try:
            # List directories in the category
            result = subprocess.run([
                "gsutil", "ls", search_path
            ], capture_output=True, text=True, check=True, timeout=45)
            
            category_trajectories = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or not line.endswith('/'):
                    continue
                    
                # Check if this is a date directory (YYYY-MM-DD format)
                dir_name = line.rstrip('/').split('/')[-1]
                if len(dir_name) == 10 and dir_name.count('-') == 2:
                    # This is a date directory, scan inside for trajectories
                    try:
                        date_result = subprocess.run([
                            "gsutil", "ls", line
                        ], capture_output=True, text=True, check=True, timeout=30)
                        
                        for traj_line in date_result.stdout.strip().split('\n'):
                            traj_line = traj_line.strip()
                            if traj_line and traj_line.endswith('/'):
                                traj_path = traj_line.rstrip('/')
                                category_trajectories.append(traj_path)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        continue
                else:
                    # Direct trajectory directory
                    traj_path = line.rstrip('/')
                    # Filter out category directories themselves
                    if not traj_path.endswith(('/success', '/failure')):
                        category_trajectories.append(traj_path)
            
            print(f"    ✅ Found {len(category_trajectories)} trajectories in {lab}/{category}")
            lab_trajectories.extend(category_trajectories)
            
            # Small delay to be nice to GCS
            time.sleep(0.1)
            
        except subprocess.CalledProcessError:
            print(f"    ⚠️  No {category} directory found in {lab}")
            continue
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Timeout scanning {lab}/{category}")
            continue
    
    return lab_trajectories


def scan_all_droid_trajectories(base_path: str = "gs://gresearch/robotics/droid_raw/1.0.1/") -> List[str]:
    """
    Comprehensively scan GCS for all available DROID trajectories using Ray parallelization.
    
    Args:
        base_path: Base GCS path to scan
        
    Returns:
        List of all trajectory GCS paths found
    """
    print(f"🔍 Comprehensive scan of {base_path}")
    print("⚡ Using Ray for parallel scanning")
    print("=" * 60)
    
    # Known lab directories 
    labs = ['AUTOLab', 'CLVR', 'GuptaLab', 'ILIAD', 'IPRL', 'IRIS', 'PennPAL', 'RAD', 'RAIL', 'REAL', 'RPL', 'TRI', 'WEIRD']
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    print(f"🚀 Launching {len(labs)} parallel scanning tasks...")
    
    # Create Ray tasks for each lab
    futures = [scan_lab_trajectories.remote(lab, base_path) for lab in labs]
    
    # Wait for all tasks to complete
    lab_results = ray.get(futures)
    
    # Combine results
    all_trajectories = []
    for lab_trajectories in lab_results:
        all_trajectories.extend(lab_trajectories)
    
    # Remove duplicates and filter
    unique_trajectories = list(set(all_trajectories))
    filtered_trajectories = []
    
    for traj in unique_trajectories:
        traj_name = traj.split('/')[-1]
        # Filter out obviously non-trajectory directories
        if (len(traj_name) > 3 and  # Reasonable length
            traj_name not in ['success', 'failure'] and  # Not category dirs
            not (len(traj_name) == 10 and traj_name.count('-') == 2)):  # Not date dirs
            filtered_trajectories.append(traj)
    
    return sorted(filtered_trajectories)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive DROID trajectory scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan all trajectories and save to file
    python scan_all_trajectories.py --output all_droid_paths.txt
    
    # Scan with custom base path
    python scan_all_trajectories.py \\
        --base-path gs://gresearch/robotics/droid_raw/1.0.1/ \\
        --output custom_paths.txt
        """)
    
    parser.add_argument(
        "--base-path",
        default="gs://gresearch/robotics/droid_raw/1.0.1/",
        help="Base GCS path to scan (default: gs://gresearch/robotics/droid_raw/1.0.1/)"
    )
    parser.add_argument(
        "--output",
        default="results/all_droid_trajectory_paths.txt",
        help="Output file for trajectory paths (default: all_droid_trajectory_paths.txt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show scan plan without actually scanning"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 Dry Run - Scan Plan")
        print("=" * 30)
        print(f"Base path: {args.base_path}")
        print(f"Output file: {args.output}")
        print("Parallelization: Ray parallel")
        print("Labs to scan: AUTOLab, CLVR, GuptaLab, ILIAD, IPRL, IRIS, PennPAL, RAD, RAIL, REAL, RPL, TRI, WEIRD")
        print("Categories: success, failure")
        return 0
    
    # Check gsutil availability
    try:
        subprocess.run(["gsutil", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gsutil not found! Please install Google Cloud SDK:")
        print("   https://cloud.google.com/sdk/docs/install")
        return 1
    
    # Scan trajectories
    start_time = time.time()
    
    try:
        trajectories = scan_all_droid_trajectories(args.base_path)
        scan_time = time.time() - start_time
        
        # Analyze results
        success_count = sum(1 for t in trajectories if 'success' in t)
        failure_count = sum(1 for t in trajectories if 'failure' in t)
        
        print(f"\n📊 Scan Complete!")
        print(f"⏱️  Total time: {scan_time/60:.1f} minutes")
        print(f"📈 Total trajectories found: {len(trajectories)}")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failure: {failure_count}")
        print(f"   ❓ Other: {len(trajectories) - success_count - failure_count}")
        
        # Save to file
        with open(args.output, 'w') as f:
            for path in trajectories:
                f.write(path + '\n')
        
        print(f"\n💾 Saved {len(trajectories)} trajectory paths to {args.output}")
        
        # Show some examples
        if trajectories:
            print(f"\n📋 Sample trajectories:")
            for i, traj in enumerate(trajectories[:5], 1):
                traj_name = traj.split('/')[-1]
                traj_type = "success" if 'success' in traj else "failure" if 'failure' in traj else "unknown"
                print(f"  {i}. {traj_name} ({traj_type})")
            if len(trajectories) > 5:
                print(f"  ... and {len(trajectories) - 5} more")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Scan interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())