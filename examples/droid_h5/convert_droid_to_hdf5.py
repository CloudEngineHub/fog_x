#!/usr/bin/env python3
"""
Convert DROID VLA trajectories to HDF5 format

This script provides a streamlined interface for converting DROID .vla files
to the new HDF5 format for use with the VLM processing pipeline.
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
import time

# Add RoboDM to path
sys.path.append('/home/syx/ucsf/robodm')

def convert_single_trajectory(input_path: str, output_path: str) -> bool:
    """Convert a single VLA trajectory to HDF5."""
    try:
        # Import here to avoid dependency issues if not available
        sys.path.append('/home/syx/ucsf/robodm/examples/droid')
        from droid_to_robodm import DROIDProcessor
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert using DROIDProcessor
        processor = DROIDProcessor()
        
        # Load DROID data (assuming VLA file is a directory for now)
        if input_path.endswith('.vla'):
            # For now, VLA files need special handling - let's skip this
            print(f"  ⚠️ VLA files not yet supported directly. Use the complete pipeline for GCS download.")
            return False
        
        droid_data = processor.load_droid_trajectory(input_path)
        
        # Convert to RoboDM format
        processor.convert_to_robodm(droid_data, output_path)
        result = True
        
        if result:
            print(f"  ✅ {os.path.basename(input_path)} → {os.path.basename(output_path)}")
            return True
        else:
            print(f"  ❌ Failed: {os.path.basename(input_path)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error converting {os.path.basename(input_path)}: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, pattern: str = "*.vla") -> tuple:
    """Convert all VLA files in a directory to HDF5."""
    
    # Find all VLA files
    search_pattern = os.path.join(input_dir, pattern)
    vla_files = glob(search_pattern)
    
    if not vla_files:
        print(f"❌ No files found matching {search_pattern}")
        return 0, 0
    
    print(f"📂 Found {len(vla_files)} VLA files to convert")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, vla_path in enumerate(vla_files, 1):
        # Generate output path
        vla_name = os.path.basename(vla_path)
        h5_name = os.path.splitext(vla_name)[0] + ".h5"
        h5_path = os.path.join(output_dir, h5_name)
        
        # Skip if output already exists
        if os.path.exists(h5_path):
            print(f"  ⏩ [{i}/{len(vla_files)}] Skipping existing: {h5_name}")
            continue
        
        print(f"  🔄 [{i}/{len(vla_files)}] Converting: {vla_name}")
        
        if convert_single_trajectory(vla_path, h5_path):
            successful += 1
        else:
            failed += 1
        
        # Progress update
        if i % 10 == 0 or i == len(vla_files):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(vla_files) - i) / rate if rate > 0 else 0
            print(f"    📊 Progress: {i}/{len(vla_files)} (Rate: {rate:.1f}/min, ETA: {eta/60:.1f}min)")
    
    return successful, failed


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert DROID VLA trajectories to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single trajectory
    python convert_droid_to_hdf5.py \\
        --input trajectory.vla \\
        --output trajectory.h5

    # Convert entire directory
    python convert_droid_to_hdf5.py \\
        --input-dir /path/to/droid/trajectories/ \\
        --output-dir /path/to/hdf5/trajectories/

    # Convert with custom pattern
    python convert_droid_to_hdf5.py \\
        --input-dir /path/to/droid/ \\
        --output-dir /path/to/hdf5/ \\
        --pattern "*_success_*.vla"
        """)
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", 
        help="Single VLA file to convert"
    )
    input_group.add_argument(
        "--input-dir", 
        help="Directory containing VLA files to convert"
    )
    
    # Output options
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output", 
        help="Output HDF5 file path (for single file conversion)"
    )
    output_group.add_argument(
        "--output-dir", 
        help="Output directory for HDF5 files (for directory conversion)"
    )
    
    # Additional options
    parser.add_argument(
        "--pattern",
        default="*.vla",
        help="File pattern to match in input directory (default: *.vla)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    
    print("🔄 DROID VLA → HDF5 Conversion")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        if args.input:
            # Single file conversion
            if not os.path.exists(args.input):
                print(f"❌ Input file not found: {args.input}")
                return 1
            
            if args.dry_run:
                print(f"Would convert: {args.input} → {args.output}")
                return 0
            
            print(f"📄 Converting single file:")
            print(f"  Input:  {args.input}")
            print(f"  Output: {args.output}")
            
            success = convert_single_trajectory(args.input, args.output)
            
            if success:
                print("✅ Conversion completed successfully!")
                return 0
            else:
                print("❌ Conversion failed!")
                return 1
        
        else:
            # Directory conversion
            if not os.path.exists(args.input_dir):
                print(f"❌ Input directory not found: {args.input_dir}")
                return 1
            
            print(f"📁 Converting directory:")
            print(f"  Input:   {args.input_dir}")
            print(f"  Output:  {args.output_dir}")
            print(f"  Pattern: {args.pattern}")
            
            if args.dry_run:
                # Show what would be converted
                search_pattern = os.path.join(args.input_dir, args.pattern)
                vla_files = glob(search_pattern)
                print(f"\nWould convert {len(vla_files)} files:")
                for vla_path in vla_files:
                    vla_name = os.path.basename(vla_path)
                    h5_name = os.path.splitext(vla_name)[0] + ".h5"
                    print(f"  {vla_name} → {h5_name}")
                return 0
            
            successful, failed = convert_directory(args.input_dir, args.output_dir, args.pattern)
            
            total_time = time.time() - start_time
            total = successful + failed
            
            print(f"\n📊 Conversion Summary:")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Total files: {total}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            if total > 0:
                print(f"  Success rate: {successful/total*100:.1f}%")
                print(f"  Average rate: {total/total_time*60:.1f} files/minute")
            
            if successful > 0:
                print(f"\n✅ Conversion completed! {successful} files converted to HDF5 format.")
                print(f"📁 Output directory: {args.output_dir}")
                
                print(f"\n🎯 Next Steps:")
                print(f"Run VLM processing on the converted files:")
                print(f"  cd /home/syx/ucsf/robodm/examples/droid_h5")
                print(f"  python simple_vlm_processing.py \\")
                print(f"    --trajectories {args.output_dir}/*.h5 \\")
                print(f"    --image-key \"observation/images/exterior_image_1_left\" \\")
                print(f"    --language-key \"metadata/language_instruction\" \\")
                print(f"    --question \"Is this trajectory successful?\" \\")
                print(f"    --output vlm_results.json")
                
                return 0
            else:
                print("❌ No files were successfully converted!")
                return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())