#!/usr/bin/env python3
"""
Complete RoboDM Training Pipeline Runner

This script runs the entire pipeline from LeRobot dataset ingestion to model training.
It demonstrates the complete workflow and can be used as a template for production usage.

Usage:
    python run_pipeline.py --dataset lerobot/pusht --num_episodes 50 --training_steps 1000
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile
import time

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lerobot_to_robodm_ingestion import LeRobotToRoboDMIngestion
from robodm_training_pipeline import RoboDMTrainingPipeline

# LeRobot imports for training
try:
    import torch
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch and/or LeRobot not available. Training will be skipped.")
    TORCH_AVAILABLE = False


def run_complete_pipeline(dataset_name: str, num_episodes: int = None, 
                         training_steps: int = 1000, batch_size: int = 32,
                         lr: float = 1e-4, output_dir: str = None,
                         robodm_data_dir: str = None, keep_robodm_data: bool = True):
    """
    Run the complete pipeline from ingestion to training.
    
    Args:
        dataset_name: LeRobot dataset name (e.g., 'lerobot/pusht')
        num_episodes: Number of episodes to convert (None for all)
        training_steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        output_dir: Output directory for trained model
        robodm_data_dir: Directory to save/load RoboDM data (None for temp)
        keep_robodm_data: Whether to keep RoboDM data after training
    
    Returns:
        dict: Results including paths and statistics
    """
    print("🚀 Starting Complete RoboDM Training Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Step 1: Data Ingestion
    print("\n📥 Step 1: Data Ingestion")
    print("-" * 40)
    
    if robodm_data_dir and Path(robodm_data_dir).exists():
        print(f"Using existing RoboDM data from: {robodm_data_dir}")
        results['robodm_data_dir'] = robodm_data_dir
        results['ingestion_time'] = 0
    else:
        print(f"Converting LeRobot dataset: {dataset_name}")
        print(f"Episodes to convert: {num_episodes if num_episodes else 'all'}")
        
        ingestion_start = time.time()
        
        # Create ingestion pipeline
        ingestion = LeRobotToRoboDMIngestion(
            dataset_name=dataset_name,
            output_dir=robodm_data_dir
        )
        
        # Run ingestion
        robodm_data_dir = ingestion.ingest(num_episodes=num_episodes)
        
        # Get conversion statistics
        stats = ingestion.get_conversion_stats()
        ingestion_time = time.time() - ingestion_start
        
        print(f"✅ Ingestion completed in {ingestion_time:.2f} seconds")
        print(f"   Trajectories: {stats['num_trajectories']}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   Output directory: {robodm_data_dir}")
        
        results['robodm_data_dir'] = robodm_data_dir
        results['ingestion_time'] = ingestion_time
        results['ingestion_stats'] = stats
    
    # Step 2: Dataset Loading and Processing
    print("\n📂 Step 2: Dataset Loading")
    print("-" * 40)
    
    loading_start = time.time()
    
    # Create training pipeline
    pipeline = RoboDMTrainingPipeline(robodm_data_dir)
    
    # Get dataset information
    training_info = pipeline.get_training_info()
    loading_time = time.time() - loading_start
    
    print(f"✅ Dataset loaded in {loading_time:.2f} seconds")
    print(f"   Dataset size: {training_info['dataset_size']} samples")
    print(f"   Trajectories: {training_info['num_trajectories']}")
    print(f"   Features: {training_info['features']}")
    
    results['loading_time'] = loading_time
    results['training_info'] = training_info
    
    # Step 3: Model Training
    print("\n🧠 Step 3: Model Training")
    print("-" * 40)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch/LeRobot not available, skipping training")
        results['training_time'] = 0
        results['training_successful'] = False
    elif training_steps == 0:
        print("⏭️  Skipping training as requested (training_steps = 0)")
        results['training_time'] = 0
        results['training_successful'] = True  # Skip is considered successful
    else:
        training_start = time.time()
        
        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Training steps: {training_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        
        # Create output directory
        if output_dir is None:
            output_dir = f"outputs/train/robodm_{dataset_name.split('/')[-1]}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get policy features and dataset stats
            policy_features = pipeline.get_policy_features()
            dataset_stats = pipeline.get_dataset_stats()
            
            
            # Create policy configuration
            cfg = DiffusionConfig(
                input_features=policy_features['input_features'],
                output_features=policy_features['output_features'],
                crop_shape=None,  # Disable cropping since our images are 96x96
                horizon=16  # Match the horizon used in RoboDM data generation
            )
            
            
            # Create and setup policy
            policy = DiffusionPolicy(cfg, dataset_stats=dataset_stats)
            policy.train()
            policy.to(device)
            
            # Use observation sequence collate function for DiffusionPolicy
            from torch.utils.data import default_collate
            
            def collate_fn(batch):
                """Collate function for DiffusionPolicy training with RoboDM data."""
                if not batch:
                    return {}
                
                # Use default collate for everything
                from torch.utils.data import default_collate
                collated = default_collate(batch)
                
                batch_size = len(batch)
                n_obs_steps = 2  # DiffusionPolicy default
                
                # Create observation sequences for DiffusionPolicy
                if 'observation.image' in collated:
                    # Images: [B, C, H, W] -> [B, T, C, H, W]
                    images = collated['observation.image']
                    # Create temporal sequence by repeating current observation
                    image_seq = images.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
                    collated['observation.image'] = image_seq
                
                if 'observation.state' in collated:
                    # States: [B, state_dim] -> [B, T, state_dim]
                    states = collated['observation.state']
                    state_seq = states.unsqueeze(1).repeat(1, n_obs_steps, 1)
                    collated['observation.state'] = state_seq
                
                if 'action' in collated:
                    # Actions: [B, horizon, action_dim] -> [B, action_dim, horizon]
                    if collated['action'].ndim == 3:
                        collated['action'] = collated['action'].transpose(1, 2)
                
                return collated
            
            dataloader = pipeline.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
            optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
            
            # Training loop
            print("Starting training loop...")
            step = 0
            done = False
            log_freq = max(1, training_steps // 20)  # Log 20 times during training
            
            while not done:
                for batch in dataloader:
                    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                            for k, v in batch.items()}
                    
                    
                    loss, _ = policy.forward(batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if step % log_freq == 0:
                        print(f"  Step {step:4d}/{training_steps}: loss = {loss.item():.4f}")
                    
                    step += 1
                    if step >= training_steps:
                        done = True
                        break
            
            # Save model
            policy.save_pretrained(output_path)
            training_time = time.time() - training_start
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            print(f"   Model saved to: {output_path}")
            print(f"   Final loss: {loss.item():.4f}")
            
            results['training_time'] = training_time
            results['training_successful'] = True
            results['output_dir'] = str(output_path)
            results['final_loss'] = loss.item()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            results['training_time'] = time.time() - training_start
            results['training_successful'] = False
            results['training_error'] = str(e)
    
    # Step 4: Cleanup
    print("\n🧹 Step 4: Cleanup")
    print("-" * 40)
    
    if not keep_robodm_data and 'robodm_data_dir' in results:
        data_dir = Path(results['robodm_data_dir'])
        if data_dir.exists() and str(data_dir).startswith('/tmp'):
            print(f"Cleaning up temporary RoboDM data: {data_dir}")
            import shutil
            shutil.rmtree(data_dir)
            results['robodm_data_cleaned'] = True
        else:
            print(f"Keeping RoboDM data: {data_dir}")
            results['robodm_data_cleaned'] = False
    else:
        print(f"Keeping RoboDM data: {results.get('robodm_data_dir', 'N/A')}")
        results['robodm_data_cleaned'] = False
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 Pipeline Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"  - Ingestion: {results.get('ingestion_time', 0):.2f}s")
    print(f"  - Loading: {results.get('loading_time', 0):.2f}s")
    print(f"  - Training: {results.get('training_time', 0):.2f}s")
    
    if results.get('training_successful'):
        if 'output_dir' in results:
            print(f"✅ Training successful! Model saved to: {results['output_dir']}")
        else:
            print("✅ Training skipped as requested")
    else:
        print("❌ Training failed or skipped")
    
    results['total_time'] = total_time
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run complete RoboDM training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 50 episodes of PushT and train for 1000 steps
  python run_pipeline.py --dataset lerobot/pusht --num_episodes 50 --training_steps 1000
  
  # Use existing RoboDM data for training
  python run_pipeline.py --robodm_data_dir ./robodm_data --training_steps 2000
  
  # Full pipeline with custom parameters
  python run_pipeline.py --dataset lerobot/xarm_lift_medium --num_episodes 100 \\
                         --training_steps 5000 --batch_size 64 --lr 5e-4 \\
                         --output_dir ./my_model
        """
    )
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                       help="LeRobot dataset name (e.g., lerobot/pusht)")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to convert (default: 50)")
    parser.add_argument("--robodm_data_dir", type=str, default=None,
                       help="Directory containing existing RoboDM data (skips ingestion)")
    
    # Training arguments
    parser.add_argument("--training_steps", type=int, default=1000,
                       help="Number of training steps (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for trained model")
    
    # Pipeline arguments
    parser.add_argument("--keep_robodm_data", action="store_true",
                       help="Keep RoboDM data after training (default: True)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only do ingestion")
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  RoboDM data dir: {args.robodm_data_dir or 'auto-generated'}")
    print(f"  Training steps: {args.training_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output dir: {args.output_dir or 'auto-generated'}")
    print(f"  Keep RoboDM data: {args.keep_robodm_data}")
    print(f"  Skip training: {args.skip_training}")
    
    # Override training steps if skipping training
    if args.skip_training:
        args.training_steps = 0
    
    # Run pipeline
    results = run_complete_pipeline(
        dataset_name=args.dataset,
        num_episodes=args.num_episodes,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        robodm_data_dir=args.robodm_data_dir,
        keep_robodm_data=args.keep_robodm_data
    )
    
    # Exit with appropriate code
    if results.get('training_successful', False) or args.skip_training:
        print("\n🎉 Pipeline executed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()