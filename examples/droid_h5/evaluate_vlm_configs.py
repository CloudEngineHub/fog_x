#!/usr/bin/env python3
"""
Evaluate VLM configurations on DROID trajectories.

Features:
- Download trajectories once, reuse across runs
- Vary number of evenly sampled frames (e.g., 4, 8, 16, 32)
- Vary passing method: 'stream' (per-frame) vs 'concat' (tiled grid)
- Vary camera video path keys (e.g., 'ext1_mp4_path', 'wrist_mp4_path')
- Save per-run outputs into distinct folders
- Produce a summary CSV of accuracy per configuration

Usage examples:
    python evaluate_vlm_configs.py \
        --paths-file results/all_droid_trajectory_paths.txt \
        --num-trajectories 50 \
        --eval-root ./eval_runs \
        --frame-counts 4 8 16 32 \
        --passing-methods stream concat \
        --video-path-keys ext1_mp4_path wrist_mp4_path

    # Or specify GCS trajectories directly
    python evaluate_vlm_configs.py \
        --trajectories gs://.../success/... gs://.../failure/... \
        --eval-root ./eval_runs

CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_VLM_CACHE_SIZE_MB=1024 python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-VL-32B-Instruct   --host 0.0.0.0   --port 30000 --tp 4 --mem-fraction-static 0.6 --chunked-prefill-size 4096
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local imports
from simple_vlm_processing import process_trajectories_parallel
from droid_pipeline import download_trajectories


def load_paths(paths_file: str) -> List[str]:
    try:
        with open(paths_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Failed to load paths from {paths_file}: {e}")
        return []


def sample_paths(paths: List[str], k: Optional[int], balance: Optional[float], seed: Optional[int]) -> List[str]:
    if seed is not None:
        random.seed(seed)
    if k is None or k <= 0 or k >= len(paths):
        return list(paths)
    if balance is None:
        return random.sample(paths, k)
    success_paths = [p for p in paths if 'success' in p.lower()]
    failure_paths = [p for p in paths if 'failure' in p.lower()]
    k_success = int(round(k * balance))
    k_failure = k - k_success
    chosen = random.sample(success_paths, min(k_success, len(success_paths)))
    chosen += random.sample(failure_paths, min(k_failure, len(failure_paths)))
    if len(chosen) < k:
        remaining = [p for p in paths if p not in chosen]
        chosen += random.sample(remaining, min(k - len(chosen), len(remaining)))
    return chosen


def infer_label_from_gcs_path(gcs_path: str) -> Optional[bool]:
    g = gcs_path.lower()
    if 'success' in g:
        return True
    if 'failure' in g:
        return False
    return None


def build_ground_truth_by_name(gcs_paths: List[str]) -> Dict[str, bool]:
    gt: Dict[str, bool] = {}
    for p in gcs_paths:
        traj_name = p.rstrip('/').split('/')[-1]
        label = infer_label_from_gcs_path(p)
        if label is not None:
            gt[traj_name] = label
    return gt


def compute_accuracy(results: Dict[str, Dict], gt_by_name: Dict[str, bool]) -> Tuple[int, int, int, float]:
    total = 0
    predicted = 0
    correct = 0
    for local_path, res in results.items():
        traj_name = os.path.basename(local_path.rstrip('/'))
        if traj_name not in gt_by_name:
            continue
        total += 1
        if not res.get('success', False):
            continue
        predicted += 1
        pred = bool(res.get('vlm_prediction', False))
        if pred == gt_by_name[traj_name]:
            correct += 1
    acc = (correct / predicted) if predicted > 0 else 0.0
    return total, predicted, correct, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM configs on DROID trajectories")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--paths-file", default="results/all_droid_trajectory_paths.txt",
                       help="File containing GCS trajectory paths")
    group.add_argument("--trajectories", nargs='+', help="GCS paths to DROID trajectory directories")

    parser.add_argument("--num-trajectories", type=int, help="Number of trajectories to sample")
    parser.add_argument("--balance", type=float, help="Success ratio target in sampling, e.g., 0.5")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for VLM")
    parser.add_argument("--eval-root", default="./eval_runs_2", help="Root folder for evaluation outputs")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials per configuration")

    parser.add_argument("--frame-counts", type=int, nargs='+', default=[2, 4, 8, 16, 32],
                        help="Frame counts to evaluate")
    parser.add_argument("--passing-methods", nargs='+', default=["stream"],
                        choices=["stream", "concat"], help="Passing methods to evaluate")
    parser.add_argument("--video-path-keys", nargs='*', default=["ext1_mp4_path"],
                        help="Video path keys from metadata (e.g., ext1_mp4_path wrist_mp4_path all). 'all' concatenates ext1_mp4_path and wrist_mp4_path. If omitted, auto-detect.")

    parser.add_argument("--language-key", default="metadata/language_instruction",
                        help="Language key to extract from HDF5 fallback")
    parser.add_argument("--question", default="Is this trajectory successful?",
                        help="VLM question")
    parser.add_argument("--use-gpt", action="store_true",
                        help="Use GPT vision API instead of local VLM")
    parser.add_argument("--gpt-api-key",
                        help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--gpt-model", default="gpt-5-2025-08-07",
                        # choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                        help="GPT model to use for vision tasks")

    args = parser.parse_args()

    # Handle GPT API key
    gpt_api_key = args.gpt_api_key or os.environ.get("OPENAI_API_KEY")
    if args.use_gpt and not gpt_api_key:
        print("❌ GPT API key required when using --use-gpt. Set --gpt-api-key or OPENAI_API_KEY environment variable.")
        return 1

    # Resolve GCS paths
    if args.trajectories:
        gcs_paths = list(args.trajectories)
    else:
        gcs_paths = load_paths(args.paths_file)
    if not gcs_paths:
        print("❌ No GCS trajectory paths provided or loaded")
        return 1

    # Sample
    gcs_paths = sample_paths(gcs_paths, args.num_trajectories, args.balance, args.seed)
    print(f"📊 Using {len(gcs_paths)} trajectories for evaluation")

    # Prepare eval root
    eval_root = Path(args.eval_root)
    runs_root = eval_root / "runs"
    downloads_root = eval_root / "droid_trajectories"
    os.makedirs(runs_root, exist_ok=True)

    # Download once
    print("\n📥 Downloading trajectories once for reuse...")
    successful_local_paths, failed = download_trajectories(gcs_paths, str(downloads_root), max_workers=args.max_workers)
    if not successful_local_paths:
        print("❌ Download failed for all trajectories")
        return 1
    print(f"✅ Downloaded {len(successful_local_paths)} trajectories; {len(failed)} failed")

    # Ground truth by traj_name
    gt_by_name = build_ground_truth_by_name(gcs_paths)
    # Persist ground truth CSV
    with open(eval_root / "ground_truth.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory_name", "label_success"])
        for name, label in sorted(gt_by_name.items()):
            writer.writerow([name, int(label)])

    # Evaluate configurations
    summary_rows = []
    configs = []
    for method in args.passing_methods:
        for n in args.frame_counts:
            if args.video_path_keys is None or len(args.video_path_keys) == 0:
                configs.append((method, n, None))
            else:
                for cam_key in args.video_path_keys:
                    # Handle 'all' option to concatenate ext1_mp4_path and wrist_mp4_path
                    if cam_key == "all":
                        configs.append((method, n, "all"))
                    else:
                        configs.append((method, n, cam_key))

    start_all = time.time()
    for (method, n, cam_key) in configs:
        run_name = f"method={method}_frames={n}" + (f"_cam={cam_key}" if cam_key else "")
        run_out_dir = runs_root / run_name
        os.makedirs(run_out_dir, exist_ok=True)

        per_trial_metrics = []

        for trial_idx in range(max(1, int(args.num_trials))):
            trial_num = trial_idx + 1
            trial_dir = run_out_dir / f"trial_{trial_num:02d}"
            os.makedirs(trial_dir, exist_ok=True)

            print(f"\n🚀 Run: {run_name} [trial {trial_num}/{args.num_trials}]")
            results = process_trajectories_parallel(
                trajectory_paths=successful_local_paths,
                question=args.question,
                max_workers=args.max_workers,
                output_dir=str(trial_dir),
                video_path_key=cam_key,
                num_frames=n,
                passing_method=method,
                concat_grid_cols=None,
                use_gpt=args.use_gpt,
                gpt_api_key=gpt_api_key,
                gpt_model=args.gpt_model
            )

            # Persist raw results per trial
            with open(trial_dir / "vlm_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            total, predicted, correct, acc = compute_accuracy(results, gt_by_name)
            print(f"📈 Trial {trial_num} accuracy: {acc:.3f} ({correct}/{predicted}) | total {total}")

            # Save per-trial metrics
            with open(trial_dir / "metrics.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["method", "frames", "camera_key", "trial", "total", "predicted", "correct", "accuracy"])
                writer.writerow([method, n, cam_key or "auto", trial_num, total, predicted, correct, f"{acc:.6f}"])

            per_trial_metrics.append({
                "trial": trial_num,
                "total": total,
                "predicted": predicted,
                "correct": correct,
                "accuracy": acc,
                "run_dir": str(trial_dir)
            })

            # Also add to overall summary (per-trial row)
            summary_rows.append({
                "method": method,
                "frames": n,
                "camera_key": cam_key or "auto",
                "trial": trial_num,
                "total": total,
                "predicted": predicted,
                "correct": correct,
                "accuracy": acc,
                "is_aggregate": False,
                "num_trials": int(args.num_trials),
                "accuracy_mean": None,
                "accuracy_variance": None,
                "run_dir": str(trial_dir)
            })

        # Aggregate across trials
        accuracies = [m["accuracy"] for m in per_trial_metrics]
        if len(accuracies) > 1:
            mean_acc = float(np.mean(accuracies))
            var_acc = float(np.var(accuracies, ddof=1))
        else:
            mean_acc = float(accuracies[0]) if accuracies else 0.0
            var_acc = 0.0

        print(f"📊 Aggregate over {len(accuracies)} trial(s): mean={mean_acc:.3f}, var={var_acc:.6f}")

        # Persist aggregate metrics JSON at config root
        aggregate_payload = {
            "method": method,
            "frames": n,
            "camera_key": cam_key or "auto",
            "num_trials": int(args.num_trials),
            "per_trial": per_trial_metrics,
            "accuracy_mean": mean_acc,
            "accuracy_variance": var_acc,
        }
        with open(run_out_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_payload, f, indent=2)

        # Write combined metrics (per-trial rows + aggregate row) at config root
        with open(run_out_dir / "metrics.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["method", "frames", "camera_key", "trial", "total", "predicted", "correct", "accuracy", "is_aggregate", "num_trials", "accuracy_mean", "accuracy_variance"])
            for m in per_trial_metrics:
                writer.writerow([method, n, cam_key or "auto", m["trial"], m["total"], m["predicted"], m["correct"], f"{m['accuracy']:.6f}", 0, int(args.num_trials), "", ""]) 
            writer.writerow([method, n, cam_key or "auto", "all", "", "", "", f"{mean_acc:.6f}", 1, int(args.num_trials), f"{mean_acc:.6f}", f"{var_acc:.6f}"])

        # Add aggregate row to overall summary
        summary_rows.append({
            "method": method,
            "frames": n,
            "camera_key": cam_key or "auto",
            "trial": "all",
            "total": None,
            "predicted": None,
            "correct": None,
            "accuracy": mean_acc,
            "is_aggregate": True,
            "num_trials": int(args.num_trials),
            "accuracy_mean": mean_acc,
            "accuracy_variance": var_acc,
            "run_dir": str(run_out_dir)
        })

    # Write overall summary
    with open(eval_root / "summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "frames", "camera_key", "trial", "total", "predicted", "correct", "accuracy", "is_aggregate", "num_trials", "accuracy_mean", "accuracy_variance", "run_dir"])
        for r in summary_rows:
            writer.writerow([
                r["method"],
                r["frames"],
                r["camera_key"],
                r.get("trial", ""),
                r.get("total", ""),
                r.get("predicted", ""),
                r.get("correct", ""),
                f"{r['accuracy']:.6f}",
                int(bool(r.get("is_aggregate", False))),
                r.get("num_trials", ""),
                f"{r['accuracy_mean']:.6f}" if r.get("accuracy_mean") is not None else "",
                f"{r['accuracy_variance']:.6f}" if r.get("accuracy_variance") is not None else "",
                r["run_dir"],
            ])

    elapsed = time.time() - start_all
    print(f"\n🎉 Evaluation complete in {elapsed/60:.1f} minutes")
    print(f"📁 Outputs in: {eval_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


