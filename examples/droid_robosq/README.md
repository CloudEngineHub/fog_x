# DROID VLM Batch Processing

This folder contains examples for batch processing DROID robot trajectories with Vision Language Models (VLM).

## Files

### `droid_download_example.py`
Downloads DROID trajectories from Google Cloud Storage with parallel processing.

**Usage:**
```bash
python droid_download_example.py --local-dir ./droid_data --num-trajectories 50
```

**Features:**
- Parallel downloads from GCS using gsutil
- Handles nested DROID directory structure 
- Configurable number of trajectories and parallel workers

### `simple_droid_vlm_example.py`
Batch processes DROID trajectories with VLM using configurable prompts and answer extraction.

**Prerequisites:**
Start qwen VLM server:
```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-32B-Instruct --host 0.0.0.0 --port 30000 --tp 4
```

**Usage Examples:**

Binary classification:
```bash
python simple_droid_vlm_example.py --data-dir ./droid_data --prompt "Is this trajectory successful?" --answer-type binary --output results.csv
```

Multiple choice:
```bash
python simple_droid_vlm_example.py --data-dir ./droid_data --prompt "What type of task is this?" --answer-type multiple_choice --choices pick place push other --output task_analysis.csv
```

Numerical scoring:
```bash
python simple_droid_vlm_example.py --data-dir ./droid_data --prompt "Rate the trajectory quality from 1-10" --answer-type number --output quality_scores.csv
```

With reasoning:
```bash
python simple_droid_vlm_example.py --data-dir ./droid_data --prompt "Is this successful?" --answer-type binary --reasoning --output detailed_results.csv
```

## Answer Types

- **`binary`**: Extracts yes/no responses
- **`number`**: Extracts numerical values
- **`multiple_choice`**: Selects from provided choices  
- **`text`**: Extracts free text (first sentence)

## Output Format

CSV with columns:
- `trajectory_path`: Path to trajectory directory
- `trajectory_name`: Trajectory identifier
- `extracted_answer`: Parsed answer based on type
- `original_answer`: Full VLM response
- `error`: Error message if processing failed

## Quick Start

1. Download trajectories:
```bash
python droid_download_example.py --local-dir ./droid_data --num-trajectories 10
```

2. Start VLM server (see prerequisites above)

3. Process with VLM:
```bash
python simple_droid_vlm_example.py --data-dir ./droid_data --prompt "Is this trajectory successful?" --answer-type binary --output results.csv
```

## Features

- ✅ Real VLM integration with qwen/sglang
- ✅ User-configurable prompts and answer extraction
- ✅ Structured CSV output
- ✅ Multiple answer type support
- ✅ Parallel processing capability
- ✅ Error handling and logging