# DROID Pipeline: End-to-End Robot Trajectory Processing with VLM

This directory contains a complete pipeline for processing robot trajectories with Vision-Language Models (VLMs), from data download to validation. The pipeline works directly with DROID raw format and includes automatic ground truth generation.

## 🎯 Overview

The pipeline consists of four main steps:
1. **Download** DROID trajectories from GCS 
2. **Generate** ground truth labels automatically from trajectory paths
3. **Process** trajectories with VLM for analysis (success/failure classification)
4. **Validate** VLM responses against ground truth data with accuracy metrics

## 📁 Files

- **`droid_pipeline.py`** - **⭐ Complete end-to-end pipeline** (main entry point)
- **`scan_all_trajectories.py`** - Generate comprehensive trajectory paths file
- **`simple_vlm_processing.py`** - Parallel VLM processing with Ray
- **`validate_vlm_responses.py`** - Validation and metrics calculation
- **`README.md`** - This documentation

## 🚀 Quick Start

### Prerequisites

1. **Install RoboDM:**
   ```bash
   cd /home/syx/ucsf/robodm
   pip install -e .
   ```

2. **Install additional dependencies:**
   ```bash
   pip install ray opencv-python h5py matplotlib
   ```

3. **Install Google Cloud SDK (for downloading DROID data):**
   ```bash
   # See https://cloud.google.com/sdk/docs/install
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

4. **Ensure VLM service is running** (see [VLM Service Setup](#vlm-service-setup))

### ⚡ **Simplest Usage (Recommended)**

The pipeline now works with intelligent defaults - just run:

```bash
# Process 30 random trajectories with all defaults
python3 droid_pipeline.py
```

This automatically:
- ✅ Loads from pre-generated trajectory paths file (`results/all_droid_trajectory_paths.txt`)
- ✅ Selects 30 random trajectories (balanced mix of success/failure)
- ✅ Downloads trajectories from GCS
- ✅ Generates ground truth labels automatically
- ✅ Processes with VLM
- ✅ Validates results and shows accuracy metrics
- ✅ Saves all outputs to `./results/`

### Custom Usage Examples

```bash
# Different number of trajectories
python3 droid_pipeline.py --num-trajectories 50

# Different output directory
python3 droid_pipeline.py --output-dir ./my_experiment

# Skip ground truth generation (if you have manual labels)
python3 droid_pipeline.py --no-generate-ground-truth

# Balance selection (70% success, 30% failure)
python3 droid_pipeline.py --balance 0.7 --seed 42

# Use auto-scan instead of pre-generated paths
python3 droid_pipeline.py --auto-scan --num-trajectories 10

# Quick test mode with sample trajectories
python3 droid_pipeline.py --auto-scan --quick-mode --num-trajectories 3
```

### 🗂️ One-Time Setup: Generate Trajectory Paths File

For faster repeated runs, first generate a comprehensive paths file:

```bash
# Scan all DROID trajectories and save paths (takes ~10-15 minutes)
python3 scan_all_trajectories.py --output results/all_droid_trajectory_paths.txt

# This creates a file with ~75,000+ trajectory paths
# Then you can use the default pipeline which loads from this file instantly
```

## 🔧 Pipeline Stages

### Stage 1: Trajectory Discovery & Selection
- **Auto-scan mode**: Scans GCS for all available trajectories
- **Paths file mode** (default): Loads from pre-generated file for speed
- **Manual mode**: Use specific trajectory GCS paths

### Stage 2: Download
- Downloads selected trajectories from GCS using `gsutil`
- Parallel downloads with progress tracking
- Automatic retry and error handling

### Stage 3: Ground Truth Generation
- Automatically extracts success/failure labels from GCS paths
- Handles lab-specific directory structures
- Creates validation-ready ground truth JSON

### Stage 4: VLM Processing  
- Processes trajectories with Vision-Language Model
- Handles both image-based and state-only trajectories
- Creates state visualizations when no images available
- Parallel processing with Ray for scalability

### Stage 5: Validation
- Compares VLM predictions against ground truth
- Calculates accuracy, precision, recall, F1 score
- Provides detailed confusion matrix and per-trajectory results

## 📊 Understanding Results

After running the pipeline, you'll get:

### 1. VLM Results (`vlm_results.json`)
```json
{
  "./results/droid_trajectories/Wed_Jan_3_16:07:12_2024/trajectory.h5": {
    "trajectory_path": "./results/droid_trajectories/Wed_Jan_3_16:07:12_2024/trajectory.h5",
    "success": true,
    "vlm_response": "Yes, this trajectory appears successful. The robot completed the manipulation task with smooth motion and proper gripper control.",
    "language_instruction": null,
    "frames_analyzed": 1,
    "total_frames": 1
  }
}
```

### 2. Ground Truth (`generated_ground_truth.json`)
```json
{
  "./results/droid_trajectories/Wed_Jan_3_16:07:12_2024": true,
  "./results/droid_trajectories/Thu_Nov_30_01:00:17_2023": false
}
```

### 3. Validation Results (`validation_results.json`)
```json
{
  "total_processed": 30,
  "validated": 30,
  "skipped": 0,
  "metrics": {
    "accuracy": 0.867,
    "precision": 0.840,
    "recall": 0.913,
    "f1": 0.875,
    "confusion_matrix": {
      "true_positive": 21,
      "true_negative": 5,
      "false_positive": 4,
      "false_negative": 0
    }
  }
}
```

### 4. Pipeline Summary (`pipeline_summary.json`)
Complete pipeline execution statistics and timing information.

## 🏗️ VLM Service Setup

The pipeline requires a VLM service to be running. You can use the RoboDM VLM service:

### Local VLM Service
```bash
# Start the VLM service (in another terminal)
cd /home/syx/ucsf/robodm
python -m robodm.agent.vlm_service --port 30000

# The service will be available at http://localhost:30000
```

### Remote VLM Service
Update the VLM configuration in `simple_vlm_processing.py`:
```python
tools_config = {
    "tools": {
        "robo2vlm": {
            "model": "Qwen/Qwen2.5-VL-32B-Instruct", 
            "base_url": "http://your-vlm-server:30000"  # Update this
        }
    }
}
```

## ⚙️ Advanced Configuration

### Custom Questions
```bash
python3 droid_pipeline.py --question "Did the robot successfully complete the manipulation task?"
python3 droid_pipeline.py --question "Rate the trajectory quality from 1-10"
python3 droid_pipeline.py --question "What went wrong in this trajectory?"
```

### Performance Tuning
```bash
# More parallel workers
python3 droid_pipeline.py --max-workers 8

# Different image/language keys  
python3 droid_pipeline.py \
    --image-key "observation/images/wrist_camera" \
    --language-key "metadata/task_description"
```

### Balanced Dataset Creation
```bash
# Create balanced dataset with specific success/failure ratio
python3 droid_pipeline.py \
    --num-trajectories 100 \
    --balance 0.6 \  # 60% success, 40% failure
    --seed 42 \      # Reproducible results
    --output-dir ./balanced_dataset
```

## 🧪 Testing

Test the pipeline with a small sample:

```bash
# Quick test with 3 trajectories
python3 droid_pipeline.py --num-trajectories 3 --dry-run

# Run actual test
python3 droid_pipeline.py --num-trajectories 3
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "No trajectories loaded from paths file"
**Solution:** Generate the paths file first:
```bash
python3 scan_all_trajectories.py --output results/all_droid_trajectory_paths.txt
```

#### 2. "gsutil not found"
**Solution:** Install Google Cloud SDK:
```bash
curl https://sdk.cloud.google.com | bash
gcloud init
```

#### 3. "VLM processing failed"
**Solution:** Ensure VLM service is running:
```bash
curl -X GET http://localhost:30000/v1/models
```

#### 4. "No valid comparisons found"
This error has been **fixed**! The pipeline now properly matches VLM results with ground truth.

### Performance Tips

1. **Use the paths file mode** (default) for faster trajectory selection
2. **Start with small samples** (`--num-trajectories 5`) for testing  
3. **Use `--dry-run`** to verify configuration before actual processing
4. **Monitor Ray dashboard** for distributed processing: `http://localhost:8265`

## 📈 Scaling Up

### For Large Experiments (100+ trajectories):

```bash
# Large balanced experiment
python3 droid_pipeline.py \
    --num-trajectories 200 \
    --balance 0.7 \
    --max-workers 8 \
    --output-dir ./large_experiment

# Process all trajectories with manual labels  
python3 droid_pipeline.py \
    --num-trajectories 1000 \
    --no-generate-ground-truth \
    --output-dir ./full_dataset
```

### Distributed Processing
```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address='head-node-ip:6379'

# Run pipeline with distributed Ray
python3 droid_pipeline.py --max-workers 16
```

## 🚦 Pipeline Status Indicators

The pipeline provides clear progress indicators:

- 🎯 **Selected trajectories** - Shows chosen trajectories with success/failure labels
- 📥 **Download progress** - Real-time download status with ETA
- 📊 **Ground truth generation** - Automatic labeling statistics  
- 🤖 **VLM processing** - Processing progress with success/failure counts
- ✅ **Validation results** - Final accuracy metrics and confusion matrix

## 🤝 Contributing

To extend the pipeline:

1. **Add new validation metrics** in `validate_vlm_responses.py`
2. **Implement custom trajectory filtering** in `droid_pipeline.py`
3. **Add new VLM models** by updating the tools configuration
4. **Create custom ground truth sources** for specialized datasets

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{droid_vlm_pipeline,
  title={DROID VLM Pipeline: Scalable Robot Trajectory Analysis},
  author={RoboDM Team},
  year={2024},
  url={https://github.com/robodm/robodm}
}
```

---

## 🎉 **Ready to Use!**

The simplest way to get started:

```bash
python3 droid_pipeline.py
```

This will process 30 trajectories end-to-end with automatic ground truth generation and validation! 🚀