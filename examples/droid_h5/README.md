# DROID HDF5 Pipeline: End-to-End Robot Trajectory Processing with VLM

This directory contains a complete pipeline for processing robot trajectories with Vision-Language Models (VLMs), from data conversion to validation. The pipeline uses the new HDF5 backend for efficient trajectory storage and parallel processing.

## 🎯 Overview

The pipeline consists of three main steps:
1. **Convert** DROID trajectories from VLA format to HDF5 format
2. **Process** trajectories with VLM for analysis (success/failure classification, task understanding, etc.)
3. **Validate** VLM responses against ground truth data

## 📁 Files

- **`droid_hdf5_pipeline.py`** - **⭐ Complete end-to-end pipeline with gsutil download**
- **`convert_droid_to_hdf5.py`** - Convert DROID VLA files to HDF5 format
- **`simple_vlm_processing.py`** - Parallel VLM processing with Ray
- **`validate_vlm_responses.py`** - Validation and metrics calculation
- **`test_pipeline.py`** - End-to-end pipeline test
- **`README.md`** - This documentation

## 🚀 Quick Start

### Prerequisites

1. **Install RoboDM with HDF5 support:**
   ```bash
   cd /home/syx/ucsf/robodm
   pip install -e .
   ```

2. **Install additional dependencies:**
   ```bash
   pip install ray opencv-python h5py
   ```

3. **Install Google Cloud SDK (for downloading DROID data):**
   ```bash
   # See https://cloud.google.com/sdk/docs/install
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

4. **Ensure VLM service is running** (see [VLM Service Setup](#vlm-service-setup))

### Complete Pipeline (Recommended)

**The easiest way is to use the complete pipeline with auto-scan:**

```bash
# Quick mode: Use pre-defined sample trajectories (fastest for testing)
python droid_hdf5_pipeline.py \
    --auto-scan --quick-mode \
    --num-trajectories 3 \
    --output-dir ./droid_hdf5_results \
    --question "Is this trajectory successful?" \
    --max-workers 2

# Full scan: Automatically discover and select from all available trajectories  
python droid_hdf5_pipeline.py \
    --auto-scan \
    --num-trajectories 10 \
    --output-dir ./droid_hdf5_results \
    --question "Is this trajectory successful?" \
    --max-workers 4

# Balanced selection (70% success, 30% failure) with reproducible results
python droid_hdf5_pipeline.py \
    --auto-scan --quick-mode \
    --num-trajectories 20 \
    --balance 0.7 \
    --seed 42 \
    --output-dir ./results \
    --question "Did the robot complete the task successfully?"
```

**Legacy manual specification:**

```bash
# Manual trajectory specification
python droid_hdf5_pipeline.py \
    --trajectories gs://gresearch/robotics/droid_raw/1.0.1/success/2023-07-21_16-18-07 \
                  gs://gresearch/robotics/droid_raw/1.0.1/failure/2023-07-21_16-27-21 \
    --output-dir ./droid_hdf5_results \
    --question "Is this trajectory successful?" \
    --max-workers 4

# Use existing HDF5 files (skip download/conversion)
python droid_hdf5_pipeline.py \
    --trajectories dummy \
    --output-dir ./existing_results \
    --skip-download \
    --question "Did the robot complete the task successfully?"
```

### Manual Step-by-Step Process

If you prefer to run each step manually:

#### Step 1: Convert DROID Data to HDF5

```bash
# Convert a single trajectory
python convert_droid_to_hdf5.py \
    --input /path/to/trajectory.vla \
    --output /path/to/output/trajectory.h5

# Convert multiple trajectories
python convert_droid_to_hdf5.py \
    --input-dir /path/to/droid/trajectories/ \
    --output-dir /path/to/hdf5/trajectories/
```

#### Step 2: Process Trajectories with VLM

```bash
# Success/failure classification
python simple_vlm_processing.py \
    --trajectories /path/to/hdf5/*.h5 \
    --image-key "observation/images/exterior_image_1_left" \
    --language-key "metadata/language_instruction" \
    --question "Is this trajectory successful?" \
    --output results.json
```

#### Step 3: Validate Results

```bash
# Validate against filename patterns (success_*, failure_*)
python validate_vlm_responses.py \
    --results results.json \
    --ground-truth-source filename \
    --output validation_results.json \
    --verbose
```

## 🔧 Detailed Usage

### VLM Processing Options

The `simple_vlm_processing.py` script supports various options:

```bash
python simple_vlm_processing.py \
    --trajectories path1.h5 path2.h5 path3.h5 \  # Individual files
    --trajectories /path/to/trajectories/*.h5 \    # Glob patterns
    --image-key "observation/images/wrist_camera" \ # Image data key
    --language-key "metadata/task_description" \   # Language instruction key
    --question "Did the robot complete the task successfully?" \ # VLM question
    --output results.json \                        # Save results to file
    --max-workers 4                               # Parallel workers (optional)
```

**Common Image Keys for DROID Data:**
- `observation/images/exterior_image_1_left` - Left exterior camera
- `observation/images/exterior_image_2_left` - Second left camera
- `observation/images/wrist_camera` - Wrist-mounted camera (if available)

**Common Language Keys:**
- `metadata/language_instruction` - Task description
- `metadata/task_description` - Alternative task description key
- `instruction` - Simple instruction key

### Validation Options

The validation script supports three ground truth sources:

#### 1. Filename-based Ground Truth
Works with files named like `success_*.h5` or `failure_*.h5`:
```bash
python validate_vlm_responses.py \
    --results results.json \
    --ground-truth-source filename
```

#### 2. Metadata-based Ground Truth
Uses a field in the trajectory metadata:
```bash
python validate_vlm_responses.py \
    --results results.json \
    --ground-truth-source metadata \
    --metadata-key "task_success"
```

#### 3. Manual Ground Truth
Uses a JSON file with manual labels:
```bash
# Create manual_labels.json:
# {
#   "trajectory1.h5": true,
#   "trajectory2.h5": false,
#   "trajectory3": true
# }

python validate_vlm_responses.py \
    --results results.json \
    --ground-truth-source manual \
    --ground-truth-file manual_labels.json
```

## 🏗️ VLM Service Setup

The pipeline requires a VLM service to be running. You can use the RoboDM VLM service:

### Option 1: Local VLM Service
```bash
# Start the VLM service
cd /home/syx/ucsf/robodm
python -m robodm.agent.vlm_service --port 8000

# The service will be available at http://localhost:8000
```

### Option 2: Remote VLM Service
Update the VLM configuration in `simple_vlm_processing.py`:
```python
tools_config = {
    "tools": {
        "robo2vlm": {
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "temperature": 0.1,
            "max_tokens": 4096,
            "context_length": 1024,
            "base_url": "http://your-vlm-server:8000"  # Add this line
        }
    }
}
```

## 📊 Understanding Results

### VLM Processing Output
```json
{
  "/path/to/trajectory.h5": {
    "trajectory_path": "/path/to/trajectory.h5",
    "success": true,
    "error": null,
    "vlm_response": "Yes, this trajectory appears to be successful. The robot successfully completed the grasping task.",
    "language_instruction": "Pick up the red cup",
    "frames_analyzed": 6,
    "total_frames": 120
  }
}
```

### Validation Output
```json
{
  "total_processed": 100,
  "validated": 95,
  "skipped": 5,
  "metrics": {
    "accuracy": 0.895,
    "precision": 0.912,
    "recall": 0.876,
    "f1": 0.894,
    "confusion_matrix": {
      "true_positive": 42,
      "true_negative": 43,
      "false_positive": 4,
      "false_negative": 6
    }
  }
}
```

## ⚠️ Important Notes

### DROID Data Compatibility

Some DROID trajectories may not have image data or may have data compatibility issues:

- **State-only trajectories**: Some DROID trajectories contain only robot state/action data without camera images
- **SVO format images**: Some trajectories use SVO format instead of MP4, which requires additional processing
- **Data type issues**: Mixed data types in trajectories may cause loading errors

**✅ Solution**: The pipeline now automatically handles state-only trajectories by creating visualizations from robot state data (actions, joint positions, cartesian position, gripper position).

### Working with State-Only Trajectories

The VLM processing script automatically detects when no images are available and creates state visualizations:

```bash
# Pipeline automatically handles state-only trajectories
python simple_vlm_processing.py \
    --trajectories /path/to/trajectories/*.h5 \
    --image-key "observation/images/exterior_image_1_left" \
    --language-key "metadata/language_instruction" \
    --question "Is this trajectory successful?"
```

When no images are found, the system:
1. Creates 4 visualizations: actions over time, joint positions, cartesian trajectory, and gripper position
2. Uses these plots as input to the VLM for analysis
3. Adjusts the VLM prompt to indicate state-based analysis

## 🛠️ Advanced Configuration

### Custom VLM Questions
Tailor questions to your specific use case:

```bash
# Success classification
--question "Is this trajectory successful?"
--question "Did the robot complete the task successfully?"

# Quality assessment
--question "Rate the quality of this trajectory from 1-10"
--question "What could be improved in this robot execution?"

# Task understanding
--question "What task is the robot performing?"
--question "Describe what happens in this trajectory"
--question "What objects is the robot interacting with?"

# Failure analysis
--question "If this trajectory failed, what was the cause?"
--question "At what point did the robot encounter difficulties?"
```

### Performance Tuning

#### Ray Configuration
```python
# In simple_vlm_processing.py, modify ray.init():
ray.init(
    num_cpus=8,           # Use 8 CPU cores
    object_store_memory=2_000_000_000  # 2GB object store
)
```

#### Batch Processing
For large datasets, process in batches:
```bash
# Process 100 trajectories at a time
find /path/to/trajectories -name "*.h5" | head -100 | xargs python simple_vlm_processing.py --trajectories --image-key "..." --language-key "..." --question "..." --output batch1.json

find /path/to/trajectories -name "*.h5" | tail -n +101 | head -100 | xargs python simple_vlm_processing.py --trajectories --image-key "..." --language-key "..." --question "..." --output batch2.json
```

## 🧪 Testing the Pipeline

Create a test dataset to verify the pipeline:

```bash
# Create test script
cat > test_pipeline.py << 'EOF'
#!/usr/bin/env python3
import tempfile
import os
import numpy as np
from robodm import Trajectory

# Create test trajectories
temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
print(f"Creating test data in {temp_dir}")

for i in range(3):
    success = i < 2  # First 2 are success, last is failure
    filename = f"{'success' if success else 'failure'}_trajectory_{i}.h5"
    traj_path = os.path.join(temp_dir, filename)
    
    traj = Trajectory(traj_path, mode="w")
    
    for t in range(10):
        # Add random action
        traj.add("action", np.random.randn(7).astype(np.float32))
        
        # Add random image
        traj.add("observation/images/exterior_image_1_left", 
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        
        # Add task instruction
        if t == 0:
            task = f"Test task {i}: {'successful' if success else 'failed'} manipulation"
            traj.add("metadata/language_instruction", task)
    
    traj.close()
    print(f"Created {filename}")

print(f"\nTest trajectories created in: {temp_dir}")
print(f"\nRun VLM processing:")
print(f'python simple_vlm_processing.py --trajectories {temp_dir}/*.h5 --image-key "observation/images/exterior_image_1_left" --language-key "metadata/language_instruction" --question "Is this trajectory successful?" --output {temp_dir}/results.json')
print(f"\nRun validation:")
print(f'python validate_vlm_responses.py --results {temp_dir}/results.json --ground-truth-source filename --output {temp_dir}/validation.json --verbose')
EOF

python test_pipeline.py
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Missing Keys Error
```
Error: Image key 'observation/images/camera1' not found
```
**Solution:** Check available keys in your trajectories:
```python
from robodm import Trajectory
traj = Trajectory("path/to/trajectory.h5", mode="r")
data = traj.load()
print("Available keys:", list(data.keys()))
traj.close()
```

#### 2. VLM Service Connection Error
```
Error: Failed to connect to VLM service
```
**Solution:** Ensure VLM service is running and accessible:
```bash
curl -X POST http://localhost:8000/health
```

#### 3. Ray Initialization Error
```
Error: Ray cluster already running
```
**Solution:** Shutdown existing Ray cluster:
```bash
ray stop
```

#### 4. HDF5 Backend Not Found
```
Error: Unknown backend 'hdf5'
```
**Solution:** Ensure the HDF5 backend is properly installed:
```python
from robodm.backend.hdf5_backend import HDF5Backend
print("HDF5 backend available")
```

### Performance Tips

1. **Use appropriate batch sizes** for your hardware
2. **Monitor memory usage** with Ray dashboard: `ray dashboard`
3. **Use SSD storage** for trajectory files when possible
4. **Optimize image resolution** if processing speed is critical

## 📈 Scaling Up

### For Large Datasets (1000+ trajectories):

1. **Use a distributed Ray cluster:**
```bash
# Head node
ray start --head --port=6379

# Worker nodes  
ray start --address='head-node-ip:6379'
```

2. **Implement checkpointing:**
```python
# Save progress periodically
if len(results) % 100 == 0:
    with open(f"checkpoint_{len(results)}.json", "w") as f:
        json.dump(results, f)
```

3. **Use data parallelism:**
```python
# Split dataset across multiple processes
dataset_chunks = np.array_split(trajectory_paths, num_workers)
```

## 🤝 Contributing

To extend this pipeline:

1. **Add new VLM models** by modifying the tools configuration
2. **Implement custom validation metrics** in `validate_vlm_responses.py`
3. **Add new ground truth sources** by extending the validation functions
4. **Optimize processing** by implementing custom Ray actors

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{robodm_hdf5_pipeline,
  title={RoboDM HDF5 Pipeline: Scalable Robot Trajectory Processing with VLMs},
  author={RoboDM Team},
  year={2024},
  url={https://github.com/robodm/robodm}
}
```