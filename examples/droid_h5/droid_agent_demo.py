#!/usr/bin/env python3
"""
DROID Agent Demo: Natural Language Dataset Processing

This demo shows how to use the Agent system with DROID trajectories,
enabling natural language queries like:
- agent.filter("trajectories that are successful")
- agent.filter("trajectories with occluded views")
- agent.map("add success probability scores")

This integrates the DROID pipeline with the RoboDM Agent system.
"""

import argparse
import os
import sys
from pathlib import Path

# Add RoboDM to path
sys.path.append('/home/syx/ucsf/robodm')

import ray
from robodm.agent import Agent
from robodm.droid_dataset import load_droid_dataset


def demo_basic_filtering():
    """Demonstrate basic filtering with DROID dataset."""
    print("🎯 DROID Agent Demo - Basic Filtering")
    print("=" * 50)
    
    # Use some downloaded trajectories from the pipeline results
    results_dir = "./results/droid_trajectories"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run droid_hdf5_pipeline.py first to download some trajectories")
        return
    
    # Load DROID dataset
    print("📦 Loading DROID dataset...")
    dataset = load_droid_dataset(results_dir)
    print(f"✅ Loaded {len(dataset)} DROID trajectories")
    
    # Create agent
    print("🤖 Creating Agent...")
    agent = Agent(dataset)
    print("✅ Agent initialized")
    
    # Show dataset info
    print(f"\n📊 Dataset Info:")
    print(f"  Total trajectories: {agent.count()}")
    
    # Sample a few trajectories to see the data structure
    print(f"\n🔍 Sample trajectory data:")
    sample = agent.take(1)[0]
    print(f"  Keys: {list(sample.keys())}")
    if "success_label" in sample:
        print(f"  Success label: {sample['success_label']}")
    if "trajectory_name" in sample:
        print(f"  Trajectory name: {sample['trajectory_name']}")
    
    # Filter for successful trajectories
    print(f"\n🎯 Filtering for successful trajectories...")
    successful = agent.filter("trajectories that are successful")
    print(f"✅ Found {successful.count()} successful trajectories")
    
    # Filter for failed trajectories  
    print(f"\n🎯 Filtering for failed trajectories...")
    failed = agent.filter("trajectories that failed or have failure in the path")
    print(f"✅ Found {failed.count()} failed trajectories")
    
    # Take some examples
    if successful.count() > 0:
        print(f"\n✅ Successful trajectory examples:")
        for i, traj in enumerate(successful.take(3)):
            print(f"  {i+1}. {traj['trajectory_name']} (success: {traj.get('success_label', 'unknown')})")
    
    if failed.count() > 0:
        print(f"\n❌ Failed trajectory examples:")
        for i, traj in enumerate(failed.take(3)):
            print(f"  {i+1}. {traj['trajectory_name']} (success: {traj.get('success_label', 'unknown')})")


def demo_advanced_filtering():
    """Demonstrate advanced filtering with loaded trajectory data."""
    print("\n🎯 DROID Agent Demo - Advanced Filtering")
    print("=" * 50)
    
    results_dir = "./results/droid_trajectories"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load DROID dataset
    dataset = load_droid_dataset(results_dir)
    agent = Agent(dataset)
    
    print(f"📦 Loaded {agent.count()} trajectories")
    
    # Load trajectory data for more detailed filtering
    print("🔄 Loading trajectory data for advanced analysis...")
    loaded_dataset = dataset.load_trajectories()
    loaded_agent = Agent(loaded_dataset)
    
    # Check what features are available
    if loaded_agent.count() > 0:
        sample = loaded_agent.take(1)[0]
        print(f"\n🔍 Available features in loaded trajectory:")
        if "features" in sample:
            features = list(sample["features"].keys())
            print(f"  Features: {features}")
            
            # Example advanced filters based on trajectory properties
            if any("language" in f for f in features):
                print(f"\n🎯 Filtering trajectories with language instructions...")
                with_language = loaded_agent.filter("trajectories that have language instructions")
                print(f"✅ Found {with_language.count()} trajectories with language instructions")
                
                if with_language.count() > 0:
                    lang_example = with_language.take(1)[0]
                    lang_feature = [f for f in features if "language" in f][0]
                    instruction = lang_example["features"].get(lang_feature, "No instruction")
                    print(f"  Example instruction: '{instruction}'")
            
            # Filter by trajectory length
            if "trajectory_length" in sample:
                print(f"\n🎯 Filtering long trajectories (>100 timesteps)...")
                long_trajs = loaded_agent.filter("trajectories that have more than 100 timesteps")
                print(f"✅ Found {long_trajs.count()} long trajectories")
                
                if long_trajs.count() > 0:
                    example = long_trajs.take(1)[0]
                    print(f"  Example length: {example.get('trajectory_length', 'unknown')} timesteps")


def demo_with_gcs_paths():
    """Demonstrate agent with GCS trajectory paths."""
    print("\n🎯 DROID Agent Demo - GCS Integration")
    print("=" * 50)
    
    # Use a small sample of GCS paths
    gcs_paths = [
        "gs://gresearch/robotics/droid_raw/1.0.1/RAIL/success/2023-04-17/Mon_Apr_17_13:20:05_2023",
        "gs://gresearch/robotics/droid_raw/1.0.1/RAIL/failure/2023-04-17/Mon_Apr_17_13:26:20_2023",
    ]
    
    print(f"📦 Creating dataset with {len(gcs_paths)} GCS trajectories...")
    
    try:
        # Create dataset (will download on demand)
        dataset = load_droid_dataset(gcs_paths, local_dir="./temp_download")
        agent = Agent(dataset)
        
        print(f"✅ Created agent with {agent.count()} trajectories")
        
        # Filter without loading (metadata only)
        print(f"\n🎯 Filtering successful trajectories (metadata only)...")
        successful = agent.filter("trajectories that are successful based on the path")
        print(f"✅ Found {successful.count()} successful trajectories")
        
        # Show examples
        for i, traj in enumerate(successful.take_all()):
            print(f"  {i+1}. {traj['trajectory_name']} -> {traj.get('success_label', 'unknown')}")
            
    except Exception as e:
        print(f"⚠️  GCS demo failed (this is expected without proper GCS setup): {e}")
        print("   This demo requires gsutil and proper GCS authentication")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="DROID Agent Demo")
    parser.add_argument("--demo", choices=["basic", "advanced", "gcs", "all"], 
                       default="all", help="Which demo to run")
    args = parser.parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        if args.demo in ["basic", "all"]:
            demo_basic_filtering()
            
        if args.demo in ["advanced", "all"]:
            demo_advanced_filtering()
            
        if args.demo in ["gcs", "all"]:
            demo_with_gcs_paths()
            
        print(f"\n🎉 DROID Agent Demo Complete!")
        print(f"💡 Key takeaways:")
        print(f"  - Agent system now works with DROID trajectories")
        print(f"  - Natural language filtering: agent.filter('trajectories that are successful')")
        print(f"  - Lazy loading: trajectories downloaded/loaded only when needed")
        print(f"  - Ray Dataset integration: parallel processing and scalability")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()