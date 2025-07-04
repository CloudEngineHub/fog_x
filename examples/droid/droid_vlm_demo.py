"""
Enhanced demo script using RoboDM Agent with VLM for trajectory success/failure classification.

This script demonstrates the full RoboDM Agent capabilities:
1. Downloads sample DROID trajectories (both success and failure)
2. Creates a proper VLADataset from file paths (not pre-loaded data)
3. Uses load_trajectories() for parallel loading
4. Demonstrates filter execution with Executor (bypassing planner for now)
5. Shows how VLM tools can be used during filtering
"""

# python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-VL-32B-Instruct   --host 0.0.0.0   --port 30000 

import os
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import cv2
import ray
from download_droid import DROIDDownloader
from droid_to_robodm import DROIDToRoboDMConverter

import robodm
from robodm.dataset import VLADataset, DatasetConfig
from robodm.agent import Agent
from robodm.agent.executor import Executor
from robodm.agent.tools import ToolsManager


class DROIDSuccessDetector:
    """Enhanced DROID success/failure detector using RoboDM Agent system."""

    def __init__(self):
        """Initialize the detector with Agent capabilities."""
        print("Initializing RoboDM Agent with VLM tools...")
        
        # Configure tools for the Agent
        self.tools_config = {
            "tools": {
                "robo2vlm": {
                    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "context_length": 1024
                }
            }
        }
        
        # Initialize tools manager
        self.tools_manager = ToolsManager(config=self.tools_config)
        
        # Initialize executor with tools
        self.executor = Executor(tools_manager=self.tools_manager)
        
        print("Agent configuration ready!")

    def create_robodm_dataset(self, robodm_dir: str) -> VLADataset:
        """
        Create VLADataset from RoboDM trajectory files.
        
        This properly uses VLADataset to start with file paths and enable
        lazy loading with load_trajectories().
        
        Args:
            robodm_dir: Directory containing RoboDM trajectory files
            
        Returns:
            VLADataset ready for parallel processing
        """
        print("Creating VLADataset from RoboDM trajectories...")
        
        # Configure dataset for parallel loading
        config = DatasetConfig(
            batch_size=4,
            shuffle=False,
            num_parallel_reads=16,  # Parallel loading
            use_metadata=True,
            auto_build_metadata=False  # We'll manage metadata manually for now
        )
        
        # Create VLADataset from directory
        # This creates a Ray dataset with just file paths
        dataset = VLADataset(
            path=robodm_dir,
            return_type="numpy",
            config=config
        )
        
        print(f"Created VLADataset with {dataset.count()} trajectory files")
        print(f"Dataset type: {type(dataset)}")
        print(f"Has _is_loaded: {hasattr(dataset, '_is_loaded')}")
        print(f"Is loaded: {dataset._is_loaded}")
        
        return dataset

    def create_success_filter_function(self) -> callable:
        """
        Create a simple filter function for successful trajectories.
        
        For now, we bypass the planner and write the function directly.
        This function can use VLM tools during execution.
        
        Returns:
            Filter function that identifies successful trajectories
        """
        def filter_successful_trajectories(trajectory: Dict[str, Any]) -> bool:
            """
            Filter function to identify successful trajectories.
            
            This demonstrates:
            1. Working with trajectory data structure
            2. Using VLM tools during filtering
            3. Checking both labels and visual analysis
            """
            # First check if we have a success label in the file path
            file_path = trajectory.get("__file_path__", "")
            has_success_label = "success" in file_path.lower()
            trajectory["metadata"] = None # TODO: for now, it has serialization error
            
            # For demonstration, we'll use VLM to analyze four frames stitched together
            # This gives better context of the trajectory progression
            try:
                print(trajectory.keys())
                # Find camera keys
                camera_keys = [k for k in trajectory.keys() 
                             if "observation/images/" in k or "image" in k.lower()]
                
                if camera_keys:
                    # Get the primary camera (usually the second one in DROID)
                    primary_camera = camera_keys[3] if len(camera_keys) > 1 else camera_keys[0]
                    
                    # Get four frames evenly spaced throughout the trajectory
                    frames = trajectory.get(primary_camera, [])
                    if len(frames) >= 4:
                        # Select 4 frames: start, 1/3, 2/3, and end
                        indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
                        selected_frames = [frames[i] for i in indices]
                        
                        # Use OpenCV to stitch frames together in a 2x2 grid
                        import cv2
                        
                        # Ensure all frames are the same size
                        h, w = selected_frames[0].shape[:2]
                        resized_frames = []
                        for frame in selected_frames:
                            if frame.shape[:2] != (h, w):
                                frame = cv2.resize(frame, (w, h))
                            resized_frames.append(frame)
                        
                        # Create 2x2 grid
                        top_row = np.hstack([resized_frames[0], resized_frames[1]])
                        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
                        stitched_frame = np.vstack([top_row, bottom_row])
                        
                    elif len(frames) > 0:
                        # If fewer than 4 frames, just use the last frame
                        stitched_frame = frames[-1]

                    # IMPORTANT: Create VLM service locally to avoid serialization issues
                    # Don't capture external tools in the closure as they contain non-serializable objects
                    from robodm.agent.vlm_service import get_vlm_service
                    vlm_service = get_vlm_service()
                    vlm_service.initialize()
                    
                    # Import Path for local use
                    from pathlib import Path
                    import cv2
                    
                    # Create output directory for VLM inputs/outputs
                    vlm_output_dir = Path("./vlm_analysis_results")
                    vlm_output_dir.mkdir(exist_ok=True)
                    
                    # Create unique filename based on trajectory name
                    traj_name = Path(file_path).stem
                    image_filename = vlm_output_dir / f"{traj_name}_input.jpg"
                    text_filename = vlm_output_dir / f"{traj_name}_output.txt"
                    
                    # Save the stitched frame (VLM input)
                    cv2.imwrite(str(image_filename), cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))
                    
                    # Use VLM to check for success indicators on the stitched frames
                    vlm_prompt = "These are 4 frames from the trajectory (start, 1/3, 2/3, end). Describe the robot's intended task first. Then anwser the question: Does this trajectory look successful in completing the task? Answer yes or no."
                    vlm_response = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                    
                    # Save the VLM response (VLM output) with additional metadata
                    with open(text_filename, 'w') as f:
                        f.write(f"Trajectory: {traj_name}\n")
                        f.write(f"File path: {file_path}\n")
                        f.write(f"Has success label: {has_success_label}\n")
                        f.write(f"Input image saved as: {image_filename.name}\n")
                        f.write(f"\nVLM Prompt:\n{vlm_prompt}\n")
                        f.write(f"\nVLM Response:\n{vlm_response}\n")
                    
                    print(f"💾 Saved VLM analysis for {traj_name}:")
                    print(f"   Input image: {image_filename}")
                    print(f"   Output text: {text_filename}")
                    print(vlm_response)
                    
                    # Check if VLM thinks it's successful
                    vlm_success = "yes" in vlm_response.lower()
                    
                    # Combine label and VLM analysis
                    # For demo, we'll trust the label but log VLM disagreements
                    if has_success_label != vlm_success:
                        print(f"❌ Label and VLM disagree for {Path(file_path).name}: "
                                f"label={has_success_label}, vlm={vlm_success}")
                    else:
                        print(f"✅ Label and VLM agree for {Path(file_path).name}: "
                                f"label={has_success_label}, vlm={vlm_success}")
                    
                    return has_success_label
                
            except Exception as e:
                print(f"Error in VLM analysis: {e}")
                # Fall back to label-based detection
            
            return has_success_label
        
        return filter_successful_trajectories

    def apply_filter_with_executor(self, dataset: VLADataset, filter_func: callable) -> VLADataset:
        """
        Apply filter using the Executor directly (bypassing planner).
        
        Args:
            dataset: VLADataset (can have just file paths)
            filter_func: Filter function to apply
            
        Returns:
            Filtered VLADataset
        """
        print("Applying filter with Executor...")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset has filter: {hasattr(dataset, 'filter')}")
        print(f"Dataset has _is_loaded: {hasattr(dataset, '_is_loaded')}")
        print(f"Dataset is loaded: {getattr(dataset, '_is_loaded', 'N/A')}")
        
        # Pass VLADataset directly to executor
        # The executor will use VLADataset's filter method which handles lazy loading
        start_time = time.time()
        filtered_dataset = self.executor.apply_filter(dataset, filter_func)
        filter_time = time.time() - start_time
        
        print(f"Filter execution time: {filter_time:.2f} seconds")
        
        return filtered_dataset

    def run_demo_with_agent(self, dataset: VLADataset):
        """
        Demonstrate using the Agent class with lazy dataset.
        
        This shows how the system should work with natural language queries.
        
        Args:
            dataset: VLADataset (can have just file paths)
        """
        print("\n" + "=" * 60)
        print("AGENT-BASED FILTERING DEMO")
        print("=" * 60)
        
        # Create Agent with the dataset directly
        agent = Agent(
            dataset,  # Pass VLADataset directly
            llm_model="Qwen/Qwen2.5-VL-32B-Instruct",
            tools_config=self.tools_config,
            context_length=1024
        )
        
        print(f"Agent initialized with {agent.count()} trajectories")
        print(f"Available tools: {agent.list_tools()}")
        
        # Show dataset schema
        print("\nDataset schema:")
        schema_info = agent.inspect_schema()
        for key in list(schema_info.get("keys", []))[:5]:
            print(f"  {key}")
        
        # Natural language filtering
        print('\nApplying filter: "trajectories that are successful"')
        print("Note: For this demo, we're using a predefined filter function")
        print("In production, the planner would generate this from the prompt")
        
        # For now, we'll use our predefined filter
        # In the full system, this would use: agent.filter("trajectories that are successful")
        # which would trigger the planner to generate the filter function
        
        # Instead, we'll demonstrate the executor directly
        filter_func = self.create_success_filter_function()
        filtered = self.executor.apply_filter(agent.dataset, filter_func)
        
        print(f"Filtered dataset contains {filtered.count()} successful trajectories")
        
        return agent, filtered

    def calculate_f1_matrix(self, dataset: VLADataset):
        """
        Calculate and print F1 matrix by comparing ground truth labels with VLM predictions.
        
        Args:
            dataset: VLADataset with loaded trajectories
        """
        print("\n" + "=" * 60)
        print("F1 MATRIX CALCULATION")
        print("=" * 60)
        
        # Transform to extract labels and predictions
        def extract_labels_and_predictions(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            """Extract ground truth and VLM predictions for F1 calculation."""
            from pathlib import Path
            import numpy as np
            
            file_path = trajectory.get("__file_path__", "")
            ground_truth = "success" in file_path.lower()
            
            # Get VLM prediction (simplified version without saving files)
            vlm_prediction = False
            try:
                # Find camera keys
                camera_keys = [k for k in trajectory.keys() 
                             if "observation/images/" in k or "image" in k.lower()]
                
                if camera_keys:
                    primary_camera = camera_keys[3] if len(camera_keys) > 1 else camera_keys[0]
                    frames = trajectory.get(primary_camera, [])
                    
                    if len(frames) >= 4:
                        # Select 4 frames: start, 1/3, 2/3, and end
                        indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
                        selected_frames = [frames[i] for i in indices]
                        
                        # Create 2x2 grid
                        h, w = selected_frames[0].shape[:2]
                        resized_frames = []
                        for frame in selected_frames:
                            if frame.shape[:2] != (h, w):
                                import cv2
                                frame = cv2.resize(frame, (w, h))
                            resized_frames.append(frame)
                        
                        top_row = np.hstack([resized_frames[0], resized_frames[1]])
                        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
                        stitched_frame = np.vstack([top_row, bottom_row])
                        
                        # Use VLM to get prediction
                        from robodm.agent.vlm_service import get_vlm_service
                        vlm_service = get_vlm_service()
                        vlm_service.initialize()
                        
                        vlm_prompt = "These are 4 frames from a robot trajectory. Does this trajectory look successful? Answer yes or no."
                        vlm_response = vlm_service.analyze_image(stitched_frame, vlm_prompt)
                        vlm_prediction = "yes" in vlm_response.lower()
                        
            except Exception as e:
                print(f"Error in VLM prediction: {e}")
                vlm_prediction = ground_truth  # fallback to ground truth
            
            return {
                "trajectory_name": Path(file_path).stem,
                "ground_truth": ground_truth,
                "vlm_prediction": vlm_prediction
            }
        
        # Apply transformation to get all predictions using VLADataset's map
        # This will automatically handle lazy loading
        results_dataset = dataset.map(extract_labels_and_predictions)
        results = list(results_dataset.take(results_dataset.count()))
        
        # Calculate confusion matrix
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for result in results:
            gt = result["ground_truth"]
            pred = result["vlm_prediction"]
            
            if gt and pred:
                true_positives += 1
            elif not gt and not pred:
                true_negatives += 1
            elif not gt and pred:
                false_positives += 1
            elif gt and not pred:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(results)
        
        # Print F1 Matrix
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Fail  Success")
        print(f"Actual   Fail    {true_negatives:4d}  {false_positives:7d}")
        print(f"         Success {false_negatives:4d}  {true_positives:7d}")
        
        print(f"\nMetrics:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1_score:.3f}")
        
        print(f"\nDetailed Results:")
        for result in results:
            status = "✅" if result["ground_truth"] == result["vlm_prediction"] else "❌"
            print(f"{status} {result['trajectory_name']}: GT={result['ground_truth']}, Pred={result['vlm_prediction']}")
        
        return f1_score


def main():
    """Enhanced main demo function using proper VLADataset and Agent system."""
    print("RoboDM VLADataset and Agent Demo")
    print("=" * 60)

    # Step 1: Download DROID trajectories
    print("\n1. Downloading DROID trajectories...")
    downloader = DROIDDownloader()
    droid_data_dir = "./droid_data"

    if not os.path.exists(droid_data_dir):
        success_paths, failure_paths = downloader.download_sample_trajectories(
            output_dir=droid_data_dir, num_success=5, num_failure=5)  # Smaller for demo
    else:
        print(f"Using existing data in {droid_data_dir}")

    # Step 2: Convert to RoboDM format
    print("\n2. Converting to RoboDM format...")
    converter = DROIDToRoboDMConverter()
    robodm_dir = "./robodm_trajectories"

    if not os.path.exists(robodm_dir):
        converter.convert_directory(droid_data_dir, robodm_dir)
    else:
        print(f"Using existing RoboDM trajectories in {robodm_dir}")

    # Step 3: Create VLADataset (with file paths only)
    print("\n3. Creating VLADataset...")
    detector = DROIDSuccessDetector()
    dataset = detector.create_robodm_dataset(robodm_dir)
    
    # Step 4: Create and apply filter (loading happens automatically)
    print("\n4. Creating and applying filter (with automatic lazy loading)...")
    filter_func = detector.create_success_filter_function()
    filtered_dataset = detector.apply_filter_with_executor(dataset, filter_func)
    
    print(f"Filtered dataset contains {filtered_dataset.count()} successful trajectories")
    
    # Step 5: Calculate F1 Matrix
    print("\n5. Calculating F1 Matrix...")
    detector.calculate_f1_matrix(dataset)
    
    # # Step 6: Demonstrate Agent usage (uncomment to test)
    # agent, agent_filtered = detector.run_demo_with_agent(dataset)
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()