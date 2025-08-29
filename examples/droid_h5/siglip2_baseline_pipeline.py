#!/usr/bin/env python3
"""
SigLIP-2 Baseline Pipeline for DROID Trajectory Analysis

This pipeline provides an alternative baseline using SigLIP-2 for ranking trajectories 
based on cosine similarity to "failure robot trajectories" with frame stitching.

Key features:
- Uses SigLIP-2 model for vision-language embedding
- Stitches frames together to create composite trajectory images
- Ranks trajectories by cosine similarity to failure reference text
- Implements cutoff mechanism based on number of failures
- Parallel processing with Ray for scalability

Algorithm:
1. Download/process DROID trajectories (reuse existing infrastructure)
2. Extract and stitch frames from trajectory videos into composite images
3. Generate SigLIP-2 embeddings for stitched images and failure reference text
4. Compute cosine similarities between trajectory embeddings and failure text
5. Rank trajectories by similarity and apply failure cutoff
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import ray
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add RoboDM to path
import sys
sys.path.append('/home/syx/ucsf/robodm')

# Import existing DROID pipeline components
from droid_pipeline import (
    download_trajectories,
    scan_droid_trajectories,
    randomly_select_trajectories,
    load_trajectories_from_file,
    get_known_sample_trajectories
)


class SigLIP2Processor:
    """SigLIP-2 model wrapper for processing stitched trajectory frames."""
    
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", device: str = "auto"):
        """Initialize SigLIP-2 model and processor."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        
        print(f"🤖 Loading SigLIP-2 model: {model_name}")
        
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            print(f"✅ SigLIP-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load SigLIP-2 model: {e}")
            print("💡 Make sure you have transformers>=4.49.0 installed:")
            print("   pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2")
            raise
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using SigLIP-2 text encoder."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # Normalize
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode single image using SigLIP-2 vision encoder."""
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # Normalize


def extract_frames_from_video(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract frames from a video file."""
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    ⚠️  Could not open video: {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return frames
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
    except Exception as e:
        print(f"    ❌ Error extracting frames from {video_path}: {e}")
    
    return frames


def stitch_frames_into_composite(frames: List[Image.Image], grid_size: Optional[Tuple[int, int]] = None, 
                               target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Stitch multiple frames into a single composite image.
    
    Args:
        frames: List of PIL Images to stitch together
        grid_size: Optional (rows, cols) for grid layout. If None, auto-calculate
        target_size: Target size for the final composite image
    
    Returns:
        Composite PIL Image
    """
    if not frames:
        # Return blank image if no frames
        return Image.new('RGB', target_size, color=(128, 128, 128))
    
    num_frames = len(frames)
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_frames))
        rows = math.ceil(num_frames / cols)
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Calculate individual frame size in the grid
    frame_width = target_size[0] // cols
    frame_height = target_size[1] // rows
    
    # Create composite image
    composite = Image.new('RGB', target_size, color=(0, 0, 0))
    
    for i, frame in enumerate(frames):
        if i >= rows * cols:
            break
            
        # Calculate position in grid
        row = i // cols
        col = i % cols
        
        # Resize frame to fit grid cell
        resized_frame = frame.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        
        # Calculate paste position
        x = col * frame_width
        y = row * frame_height
        
        # Paste frame into composite
        composite.paste(resized_frame, (x, y))
    
    return composite


def find_trajectory_videos(trajectory_path: str) -> List[str]:
    """Find all video files in a trajectory directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for root, dirs, files in os.walk(trajectory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files


@ray.remote(num_cpus=1, num_gpus=0.1 if torch.cuda.is_available() else 0)
class SigLIP2Worker:
    """Ray worker for parallel SigLIP-2 processing with frame stitching."""
    
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        self.processor = SigLIP2Processor(model_name)
        
        # Pre-compute failure reference embedding
        self.failure_text = "This is a photo of a failed robot trajectory with errors and unsuccessful task completion."
        self.failure_embedding = self.processor.encode_text(self.failure_text)
    
    def process_trajectory(self, trajectory_path: str, max_frames_per_video: int = 8,
                          frames_per_composite: int = 16) -> Tuple[str, Dict]:
        """Process a single trajectory by stitching frames and computing similarity to failure reference."""
        try:
            trajectory_name = os.path.basename(trajectory_path)
            print(f"    🔍 Processing: {trajectory_name}")
            
            # Find video files in trajectory
            video_files = find_trajectory_videos(trajectory_path)
            
            if not video_files:
                return trajectory_path, {
                    "trajectory_path": trajectory_path,
                    "error": "No video files found",
                    "similarity_score": 0.0,
                    "frames_processed": 0
                }
            
            # Collect frames from all videos
            all_frames = []
            for video_path in video_files[:3]:  # Limit to first 3 videos
                frames = extract_frames_from_video(video_path, max_frames_per_video)
                all_frames.extend(frames)
            
            if not all_frames:
                return trajectory_path, {
                    "trajectory_path": trajectory_path,
                    "error": "No frames extracted",
                    "similarity_score": 0.0,
                    "frames_processed": 0
                }
            
            # Limit total frames and stitch into composite
            frames_to_use = all_frames[:frames_per_composite]
            composite_image = stitch_frames_into_composite(frames_to_use)
            
            # Get embedding for stitched composite
            composite_embedding = self.processor.encode_image(composite_image)
            
            # Compute cosine similarity with failure reference
            similarity = cosine_similarity(
                composite_embedding, 
                self.failure_embedding
            )
            
            similarity_score = float(similarity.cpu().numpy()[0])
            
            result = {
                "trajectory_path": trajectory_path,
                "similarity_score": similarity_score,
                "frames_processed": len(frames_to_use),
                "videos_processed": len(video_files),
                "composite_grid_size": f"{math.ceil(math.sqrt(len(frames_to_use)))}x{math.ceil(math.sqrt(len(frames_to_use)))}"
            }
            
            print(f"    ✅ {trajectory_name}: score={similarity_score:.3f}, frames={len(frames_to_use)}")
            return trajectory_path, result
            
        except Exception as e:
            error_msg = f"Error processing {trajectory_path}: {e}"
            print(f"    ❌ {error_msg}")
            return trajectory_path, {
                "trajectory_path": trajectory_path,
                "error": error_msg,
                "similarity_score": 0.0,
                "frames_processed": 0
            }


def process_trajectories_with_siglip2(
    trajectory_paths: List[str],
    model_name: str = "google/siglip2-base-patch16-224",
    max_workers: int = 4,
    max_frames_per_video: int = 8,
    frames_per_composite: int = 16
) -> Dict[str, Dict]:
    """Process trajectories using SigLIP-2 with frame stitching and compute failure similarity scores."""
    
    print(f"🤖 Processing {len(trajectory_paths)} trajectories with SigLIP-2 + Frame Stitching")
    print(f"   Model: {model_name}")
    print(f"   Max workers: {max_workers}")
    print(f"   Max frames per video: {max_frames_per_video}")
    print(f"   Frames per composite: {frames_per_composite}")
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()
    
    # Create worker pool
    workers = [SigLIP2Worker.remote(model_name) for _ in range(max_workers)]
    
    # Submit tasks to workers
    futures = []
    for i, trajectory_path in enumerate(trajectory_paths):
        worker = workers[i % max_workers]
        future = worker.process_trajectory.remote(
            trajectory_path, max_frames_per_video, frames_per_composite
        )
        futures.append(future)
    
    # Collect results
    results = {}
    completed = 0
    start_time = time.time()
    
    while futures:
        # Wait for at least one task to complete
        ready, futures = ray.wait(futures, num_returns=1, timeout=60.0)
        
        for future in ready:
            try:
                trajectory_path, result = ray.get(future)
                results[trajectory_path] = result
                completed += 1
                
                # Progress update
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(trajectory_paths) - completed) / rate if rate > 0 else 0
                
                status = "✅" if "error" not in result else "❌"
                traj_name = os.path.basename(trajectory_path)
                score = result.get("similarity_score", 0.0)
                
                print(f"{status} [{completed}/{len(trajectory_paths)}] {traj_name} "
                      f"(score: {score:.3f}, rate: {rate:.1f}/min, ETA: {eta/60:.1f}min)")
                
            except Exception as e:
                print(f"❌ Failed to get result: {e}")
                completed += 1
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - successful
    
    print(f"\n📊 SigLIP-2 Processing Summary:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Rate: {len(trajectory_paths)/total_time*60:.1f} trajectories/minute")
    
    return results


def rank_trajectories_by_failure_similarity(
    results: Dict[str, Dict],
    failure_cutoff_ratio: float = 0.3
) -> Tuple[List[Tuple[str, float]], int]:
    """
    Rank trajectories by similarity to failure reference and determine cutoff.
    
    Args:
        results: SigLIP-2 processing results
        failure_cutoff_ratio: Ratio of trajectories to classify as failures (0.0-1.0)
    
    Returns:
        Tuple of (ranked_trajectories, failure_cutoff_index)
    """
    
    # Extract valid results with similarity scores
    valid_results = [
        (traj_path, data["similarity_score"])
        for traj_path, data in results.items()
        if "error" not in data and "similarity_score" in data
    ]
    
    # Sort by similarity score (descending - higher similarity to failure = more likely failure)
    ranked_trajectories = sorted(valid_results, key=lambda x: x[1], reverse=True)
    
    # Calculate cutoff index based on failure ratio
    failure_cutoff_index = int(len(ranked_trajectories) * failure_cutoff_ratio)
    
    print(f"📊 Trajectory Ranking Summary:")
    print(f"   Total valid trajectories: {len(ranked_trajectories)}")
    print(f"   Failure cutoff ratio: {failure_cutoff_ratio:.1%}")
    print(f"   Trajectories classified as failures: {failure_cutoff_index}")
    print(f"   Trajectories classified as successes: {len(ranked_trajectories) - failure_cutoff_index}")
    
    if ranked_trajectories:
        print(f"   Similarity score range: {ranked_trajectories[-1][1]:.3f} to {ranked_trajectories[0][1]:.3f}")
        print(f"   Failure threshold score: {ranked_trajectories[failure_cutoff_index-1][1]:.3f}" if failure_cutoff_index > 0 else "N/A")
    
    return ranked_trajectories, failure_cutoff_index


def generate_baseline_predictions(
    ranked_trajectories: List[Tuple[str, float]],
    failure_cutoff_index: int,
    output_dir: str
) -> str:
    """Generate baseline predictions based on SigLIP-2 similarity ranking."""
    
    predictions = {}
    
    for i, (traj_path, similarity_score) in enumerate(ranked_trajectories):
        # Predict as failure if above cutoff threshold
        is_failure = i < failure_cutoff_index
        
        # Convert to relative path format consistent with ground truth
        output_dir_name = os.path.basename(output_dir.rstrip('/'))
        traj_name = os.path.basename(traj_path)
        relative_path = f"./{output_dir_name}/droid_trajectories/{traj_name}"
        
        predictions[relative_path] = {
            "trajectory_path": relative_path,
            "predicted_failure": is_failure,
            "success": not is_failure,  # For compatibility with validation
            "similarity_score": similarity_score,
            "rank": i + 1,
            "method": "siglip2_stitched_baseline"
        }
    
    # Save predictions
    predictions_file = os.path.join(output_dir, "siglip2_baseline_predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    failure_count = sum(1 for p in predictions.values() if p["predicted_failure"])
    success_count = len(predictions) - failure_count
    
    print(f"📊 Baseline Predictions Generated:")
    print(f"   Predicted failures: {failure_count}")
    print(f"   Predicted successes: {success_count}")
    print(f"   💾 Saved to: {predictions_file}")
    
    return predictions_file


def run_siglip2_baseline_pipeline(
    trajectory_gcs_paths: List[str],
    output_dir: str,
    model_name: str = "google/siglip2-base-patch16-224",
    failure_cutoff_ratio: float = 0.3,
    max_workers: int = 4,
    max_frames_per_video: int = 8,
    frames_per_composite: int = 16,
    skip_download: bool = False
) -> Dict:
    """
    Run complete SigLIP-2 baseline pipeline with frame stitching.
    
    Args:
        trajectory_gcs_paths: GCS paths to DROID trajectories
        output_dir: Output directory for all files
        model_name: SigLIP-2 model name to use
        failure_cutoff_ratio: Ratio of trajectories to classify as failures
        max_workers: Maximum parallel workers
        max_frames_per_video: Maximum frames to extract per video
        frames_per_composite: Maximum frames to include in stitched composite
        skip_download: Skip download if trajectories already exist locally
        
    Returns:
        Dictionary with comprehensive pipeline results
    """
    print("🎯 SigLIP-2 Baseline Pipeline - Stitched Frame Analysis")
    print("=" * 60)
    
    pipeline_start = time.time()
    trajectories_dir = os.path.join(output_dir, "droid_trajectories")
    
    results = {
        "input_trajectories": len(trajectory_gcs_paths),
        "model_name": model_name,
        "failure_cutoff_ratio": failure_cutoff_ratio,
        "frames_per_composite": frames_per_composite,
        "stages": {}
    }
    
    # Stage 1: Download DROID trajectories (reuse existing infrastructure)
    if skip_download:
        print("⏩ Skipping download - using existing DROID trajectories")
        local_paths = [d for d in Path(trajectories_dir).iterdir() if d.is_dir()]
        successful_paths = [str(p) for p in local_paths]
        failed_downloads = []
    else:
        print("\n📥 Stage 1: Download DROID Trajectories")
        print("-" * 40)
        successful_paths, failed_downloads = download_trajectories(
            trajectory_gcs_paths, trajectories_dir, max_workers
        )
    
    results["stages"]["download"] = {
        "successful": len(successful_paths),
        "failed": len(failed_downloads) if not skip_download else 0,
        "local_paths": successful_paths
    }
    
    if not successful_paths:
        print("❌ No trajectories were successfully downloaded!")
        return results
    
    # Stage 2: SigLIP-2 Processing with Frame Stitching
    print("\n🎨 Stage 2: SigLIP-2 Processing with Frame Stitching")
    print("-" * 50)
    
    try:
        siglip2_results = process_trajectories_with_siglip2(
            successful_paths,
            model_name=model_name,
            max_workers=max_workers,
            max_frames_per_video=max_frames_per_video,
            frames_per_composite=frames_per_composite
        )
        
        # Save detailed results
        siglip2_file = os.path.join(output_dir, "siglip2_detailed_results.json")
        with open(siglip2_file, 'w') as f:
            json.dump(siglip2_results, f, indent=2)
        
        results["stages"]["siglip2_processing"] = {
            "total_processed": len(siglip2_results),
            "successful": sum(1 for r in siglip2_results.values() if "error" not in r),
            "failed": sum(1 for r in siglip2_results.values() if "error" in r),
            "results_file": siglip2_file
        }
        
    except Exception as e:
        print(f"❌ SigLIP-2 processing failed: {e}")
        return results
    
    # Stage 3: Ranking and Classification
    print("\n📊 Stage 3: Trajectory Ranking & Classification")
    print("-" * 50)
    
    ranked_trajectories, failure_cutoff_index = rank_trajectories_by_failure_similarity(
        siglip2_results, failure_cutoff_ratio
    )
    
    results["stages"]["ranking"] = {
        "total_ranked": len(ranked_trajectories),
        "predicted_failures": failure_cutoff_index,
        "predicted_successes": len(ranked_trajectories) - failure_cutoff_index,
        "failure_threshold_score": ranked_trajectories[failure_cutoff_index-1][1] if failure_cutoff_index > 0 else None
    }
    
    # Stage 4: Generate Baseline Predictions
    print("\n📋 Stage 4: Generate Baseline Predictions")
    print("-" * 45)
    
    predictions_file = generate_baseline_predictions(
        ranked_trajectories, failure_cutoff_index, output_dir
    )
    
    results["stages"]["predictions"] = {
        "predictions_file": predictions_file,
        "predicted_failures": failure_cutoff_index,
        "predicted_successes": len(ranked_trajectories) - failure_cutoff_index
    }
    
    # Pipeline Summary
    total_time = time.time() - pipeline_start
    results["total_time"] = total_time
    
    print(f"\n🎉 SigLIP-2 Baseline Pipeline Complete!")
    print(f"📊 Total time: {total_time/60:.1f} minutes")
    print(f"📁 All results saved to: {output_dir}")
    
    # Save pipeline summary
    summary_file = os.path.join(output_dir, "siglip2_baseline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Pipeline summary: {summary_file}")
    print(f"🔍 Predictions file: {predictions_file}")
    print(f"📊 Use validate_vlm_responses.py to compare against ground truth")
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="SigLIP-2 Baseline Pipeline with Frame Stitching for DROID Trajectory Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Use pre-generated paths with SigLIP-2 baseline
    python siglip2_baseline_pipeline.py
    
    # Custom model and parameters
    python siglip2_baseline_pipeline.py \\
        --model-name google/siglip2-so400m-patch14-224 \\
        --failure-cutoff-ratio 0.4 \\
        --frames-per-composite 20 \\
        --num-trajectories 50
    
    # Quick test mode
    python siglip2_baseline_pipeline.py \\
        --auto-scan --quick-mode \\
        --num-trajectories 10 \\
        --frames-per-composite 8
        """)
    
    # Trajectory selection arguments
    trajectory_group = parser.add_mutually_exclusive_group(required=False)
    trajectory_group.add_argument(
        "--trajectories", nargs="+",
        help="GCS paths to DROID trajectory directories"
    )
    trajectory_group.add_argument(
        "--auto-scan", action="store_true",
        help="Auto-scan GCS for trajectories"
    )
    trajectory_group.add_argument(
        "--paths-file", default="results/all_droid_trajectory_paths.txt",
        help="Load trajectory paths from file"
    )
    
    parser.add_argument(
        "--num-trajectories", type=int, default=100,
        help="Number of trajectories to select (default: 30)"
    )
    parser.add_argument(
        "--balance", type=float,
        help="Success/failure balance for selection (0.0-1.0)"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for reproducible selection"
    )
    
    # SigLIP-2 specific arguments
    parser.add_argument(
        "--model-name", default="google/siglip2-base-patch16-224",
        help="SigLIP-2 model name (default: google/siglip2-base-patch16-224)"
    )
    parser.add_argument(
        "--failure-cutoff-ratio", type=float, default=0.3,
        help="Ratio of trajectories to classify as failures (default: 0.3)"
    )
    parser.add_argument(
        "--max-frames-per-video", type=int, default=8,
        help="Max frames to extract per video (default: 8)"
    )
    parser.add_argument(
        "--frames-per-composite", type=int, default=16,
        help="Max frames to include in stitched composite (default: 16)"
    )
    
    # General arguments
    parser.add_argument(
        "--output-dir", default="./siglip2_baseline_output",
        help="Output directory (default: ./siglip2_baseline_output)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Max parallel workers (default: 4)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use existing trajectories"
    )
    parser.add_argument(
        "--base-path", default="gs://gresearch/robotics/droid_raw/1.0.1/",
        help="Base GCS path for auto-scan"
    )
    parser.add_argument(
        "--quick-mode", action="store_true",
        help="Use pre-defined sample trajectories for testing"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without running"
    )
    
    args = parser.parse_args()
    
    # Handle trajectory selection
    if args.trajectories:
        trajectory_paths = args.trajectories
    elif args.auto_scan:
        all_trajectories = scan_droid_trajectories(args.base_path, args.quick_mode)
        if not all_trajectories:
            print("❌ No trajectories found!")
            return 1
        trajectory_paths = randomly_select_trajectories(
            all_trajectories, args.num_trajectories, args.balance, args.seed
        )
    else:
        all_trajectories = load_trajectories_from_file(args.paths_file)
        if not all_trajectories:
            print("❌ No trajectories loaded from paths file!")
            return 1
        trajectory_paths = randomly_select_trajectories(
            all_trajectories, args.num_trajectories, args.balance, args.seed
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dry_run:
        print("🔍 SigLIP-2 Stitched Baseline - Configuration")
        print("=" * 50)
        print(f"Model: {args.model_name}")
        print(f"Failure cutoff ratio: {args.failure_cutoff_ratio}")
        print(f"Max frames per video: {args.max_frames_per_video}")
        print(f"Frames per composite: {args.frames_per_composite}")
        print(f"Selected trajectories: {len(trajectory_paths)}")
        print(f"Output directory: {args.output_dir}")
        return 0
    
    try:
        results = run_siglip2_baseline_pipeline(
            trajectory_gcs_paths=trajectory_paths,
            output_dir=args.output_dir,
            model_name=args.model_name,
            failure_cutoff_ratio=args.failure_cutoff_ratio,
            max_workers=args.max_workers,
            max_frames_per_video=args.max_frames_per_video,
            frames_per_composite=args.frames_per_composite,
            skip_download=args.skip_download
        )
        
        print(f"\n🎉 SigLIP-2 Baseline Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    exit(main())