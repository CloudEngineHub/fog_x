"""
Enhanced demo script using RoboDM Agent with VLM for trajectory success/failure classification.

This script demonstrates the full RoboDM Agent capabilities:
1. Downloads sample DROID trajectories (both success and failure)
2. Creates a proper VLADataset from file paths (not pre-loaded data)
3. Uses load_trajectories() for parallel loading
4. Demonstrates filter execution with Executor (bypassing planner for now)
5. Shows how VLM tools can be used during filtering
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
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
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "temperature": 0.1,
                    "max_tokens": 100,
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
        
        return dataset

    def load_and_materialize_dataset(self, dataset: VLADataset) -> VLADataset:
        """
        Load trajectories in parallel and materialize the dataset.
        
        This demonstrates the proper use of load_trajectories() for
        parallel data loading.
        
        Args:
            dataset: VLADataset with file paths
            
        Returns:
            VLADataset with loaded trajectory data
        """
        print("Loading trajectories in parallel...")
        
        # Load trajectories - this transforms file paths to actual data
        # The loading happens in parallel across Ray workers
        loaded_dataset = dataset.load_trajectories()
        
        # Materialize to ensure data is computed and cached
        print("Materializing dataset...")
        loaded_dataset.materialize()
        
        print(f"Loaded and materialized {loaded_dataset.count()} trajectories")
        
        return loaded_dataset

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
            
            # For demonstration, we'll also use VLM to analyze the last frame
            # In a real scenario, you might want more sophisticated logic
            try:
                # Find camera keys
                camera_keys = [k for k in trajectory.keys() 
                             if "observation/images/" in k or "image" in k.lower()]
                
                if camera_keys:
                    # Get the primary camera (usually the second one in DROID)
                    primary_camera = camera_keys[1] if len(camera_keys) > 1 else camera_keys[0]
                    
                    # Get the last frame
                    frames = trajectory.get(primary_camera, [])
                    if len(frames) > 0:
                        last_frame = frames[-1]
                        
                        # IMPORTANT: Create VLM tool locally inside the function
                        # This avoids capturing it in the closure which would cause serialization issues
                        from robodm.agent.vlm_service import get_vlm_service
                        vlm_service = get_vlm_service()
                        vlm_service.initialize()
                        
                        # Use VLM to check for success indicators
                        vlm_response = vlm_service.analyze_image(
                            last_frame, 
                            "Is this robot task completed successfully? Answer yes or no."
                        )
                        
                        # Check if VLM thinks it's successful
                        vlm_success = "yes" in vlm_response.lower()
                        
                        # Combine label and VLM analysis
                        # For demo, we'll trust the label but log VLM disagreements
                        if has_success_label != vlm_success:
                            print(f"Label and VLM disagree for {Path(file_path).name}: "
                                  f"label={has_success_label}, vlm={vlm_success}")
                        
                        return has_success_label
                
            except Exception as e:
                print(f"Error in VLM analysis: {e}")
                # Fall back to label-based detection
            
            return has_success_label
        
        return filter_successful_trajectories

    def apply_filter_with_executor(self, dataset: VLADataset, filter_func: callable) -> ray.data.Dataset:
        """
        Apply filter using the Executor directly (bypassing planner).
        
        Args:
            dataset: VLADataset with loaded trajectories
            filter_func: Filter function to apply
            
        Returns:
            Filtered Ray dataset
        """
        print("Applying filter with Executor...")
        
        # Get the underlying Ray dataset
        ray_dataset = dataset.get_ray_dataset()
        
        # Apply filter using executor
        start_time = time.time()
        filtered_dataset = self.executor.apply_filter(ray_dataset, filter_func)
        filter_time = time.time() - start_time
        
        print(f"Filter execution time: {filter_time:.2f} seconds")
        
        return filtered_dataset

    def analyze_results(self, original_dataset: VLADataset, filtered_dataset: ray.data.Dataset):
        """
        Analyze and display results of the filtering operation.
        
        Args:
            original_dataset: Original VLADataset
            filtered_dataset: Filtered Ray dataset
        """
        print("\n" + "=" * 60)
        print("FILTERING RESULTS")
        print("=" * 60)
        
        # Get counts
        total_count = original_dataset.count()
        success_count = filtered_dataset.count()
        
        print(f"Total trajectories: {total_count}")
        print(f"Filtered (successful): {success_count}")
        print(f"Filtered (failed): {total_count - success_count}")
        
        # Sample analysis of filtered trajectories
        if success_count > 0:
            print("\nAnalyzing sample successful trajectory...")
            sample = filtered_dataset.take(1)[0]
            
            # Show trajectory info
            file_path = sample.get("__file_path__", "unknown")
            print(f"Sample trajectory: {Path(file_path).name}")
            
            # Find available data keys
            data_keys = [k for k in sample.keys() if not k.startswith("__")]
            print(f"Available data keys: {data_keys[:5]}...")  # Show first 5
            
            # Check trajectory length
            if data_keys:
                first_key = data_keys[0]
                if hasattr(sample[first_key], "__len__"):
                    print(f"Trajectory length: {len(sample[first_key])} frames")

    def run_demo_with_agent(self, loaded_dataset: VLADataset):
        """
        Demonstrate using the Agent class with proper dataset.
        
        This shows how the system should work with natural language queries.
        
        Args:
            loaded_dataset: VLADataset with loaded trajectories
        """
        print("\n" + "=" * 60)
        print("AGENT-BASED FILTERING DEMO")
        print("=" * 60)
        
        # Create Agent with the loaded dataset
        agent = Agent(
            loaded_dataset.get_ray_dataset(),
            llm_model="Qwen/Qwen2.5-VL-7B-Instruct",
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
    
    # Step 4: Load trajectories in parallel
    print("\n4. Loading trajectories in parallel...")
    loaded_dataset = detector.load_and_materialize_dataset(dataset)
    
    # Step 5: Create and apply filter
    print("\n5. Creating and applying filter...")
    filter_func = detector.create_success_filter_function()
    filtered_dataset = detector.apply_filter_with_executor(loaded_dataset, filter_func)
    
    # Step 6: Analyze results
    detector.analyze_results(loaded_dataset, filtered_dataset)
    
    # Step 7: Demonstrate Agent usage
    agent, agent_filtered = detector.run_demo_with_agent(loaded_dataset)
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()

    print("\nDemo completed successfully!")
    print("Key improvements demonstrated:")
    print("- VLADataset created from file paths (not pre-loaded data)")
    print("- Parallel loading with load_trajectories()")
    print("- Filter execution with Executor")
    print("- VLM tool usage during filtering")
    print("- Proper dataset materialization")


if __name__ == "__main__":
    main()