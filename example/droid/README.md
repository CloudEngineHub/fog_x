# DROID Trajectory Analysis with RoboDM

This example demonstrates how to download DROID trajectories, convert them to RoboDM format, and use the robo2vlm vision-language model to analyze success/failure patterns.

## Files

- `download_droid.py`: Downloads sample DROID trajectories from Google Cloud Storage
- `droid_to_robodm.py`: Converts DROID trajectories to RoboDM VLA format
- `droid_vlm_demo.py`: Uses robo2vlm to analyze trajectories and classify success/failure

## Usage

Run the complete demo:

```bash
python droid_vlm_demo.py
```

This will:
1. Download 2 successful and 2 failed DROID trajectories
2. Convert them to RoboDM format (.vla files)
3. Use the robo2vlm tool to analyze frames and detect success/failure patterns
4. Report classification accuracy

## Individual Scripts

### Download DROID trajectories only:
```bash
python download_droid.py
```

### Convert existing DROID data to RoboDM:
```bash
python droid_to_robodm.py
```

## Requirements

- gsutil (for downloading from Google Cloud Storage)
- RoboDM with vision tools enabled
- VLM model (qwen2.5-7b by default)

## Sample Output

The demo will show:
- Frame-by-frame analysis of robot tasks
- Success/failure indicators detected by VLM
- Overall trajectory classification accuracy
- Common task descriptions extracted from visual data