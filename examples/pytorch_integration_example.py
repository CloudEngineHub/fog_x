"""
Example: Using the new ingestion API with PyTorch datasets.

This example shows how users can quickly convert their existing PyTorch
datasets into VLA datasets with minimal code changes.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch

from robodm.ingestion import (PyTorchDatasetAdapter,
                              create_vla_dataset_from_source)


# Example PyTorch dataset (simulating existing user code)
class CustomVisionDataset(torch.utils.data.Dataset):
    """Example PyTorch dataset for computer vision tasks."""

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate image and label data
        image = torch.randn(3, 224, 224)  # RGB image
        label = torch.randint(0, 10, (1, )).item()  # Classification label
        metadata = {"idx": idx, "source": "synthetic"}

        return image, label, metadata


class CustomTimeSeriesDataset(torch.utils.data.Dataset):
    """Example PyTorch dataset for time series data."""

    def __init__(self, num_samples: int = 500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate time series data
        sequence_length = 100
        num_features = 10

        data = torch.randn(sequence_length, num_features)
        target = torch.randn(1)

        return {
            "sequence": data,
            "target": target,
            "timestamp": idx * 0.1,  # 0.1 second intervals
            "metadata": {
                "patient_id": f"patient_{idx % 50}"
            },
        }


# Example 1: Simple conversion with automatic detection
def example_simple_pytorch_conversion():
    """Convert PyTorch dataset to VLA dataset with minimal code."""

    # Create your existing PyTorch dataset
    pytorch_dataset = CustomVisionDataset(num_samples=1000)

    # Convert to VLA dataset with one line of code!
    vla_dataset = create_vla_dataset_from_source(
        data_source=pytorch_dataset,
        output_directory="./vision_trajectories",
        num_workers=4,
    )

    print(f"Created VLA dataset with {vla_dataset.count()} items")
    return vla_dataset


# Example 2: Custom transformation function
def example_pytorch_with_transform():
    """Convert PyTorch dataset with custom transformation."""

    def transform_vision_data(data_tuple):
        """Transform PyTorch dataset output into robodm format."""
        image, label, metadata = data_tuple

        # Convert torch tensors to numpy (robodm-friendly format)
        return {
            "image": image.numpy().transpose(1, 2, 0),  # CHW -> HWC
            "label": label,
            "metadata": metadata,
            "image_stats": {
                "mean": float(image.mean()),
                "std": float(image.std())
            },
        }

    pytorch_dataset = CustomVisionDataset(num_samples=1000)

    vla_dataset = create_vla_dataset_from_source(
        data_source=pytorch_dataset,
        transform_fn=transform_vision_data,
        output_directory="./vision_transformed_trajectories",
        num_workers=4,
        group_size=100,  # 100 images per trajectory file
    )

    return vla_dataset


# Example 3: Time series data with automatic handling
def example_timeseries_pytorch():
    """Convert time series PyTorch dataset."""

    # Time series dataset that already returns dicts
    pytorch_dataset = CustomTimeSeriesDataset(num_samples=500)

    # VLA dataset will automatically handle dict outputs
    vla_dataset = create_vla_dataset_from_source(
        data_source=pytorch_dataset,
        output_directory="./timeseries_trajectories",
        num_workers=2,
        group_size=50,  # 50 sequences per trajectory
    )

    return vla_dataset


# Example 4: Manual adapter usage for more control
def example_manual_adapter():
    """Use adapter manually for more control over the process."""

    def custom_transform(data_tuple):
        """Custom transformation with validation."""
        image, label, metadata = data_tuple

        # Add validation
        if image.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[0]}")

        # Custom processing
        image_np = image.numpy().transpose(1, 2, 0)

        # Normalize to 0-255 range for better visualization
        image_np = ((image_np - image_np.min()) /
                    (image_np.max() - image_np.min()) * 255).astype(np.uint8)

        return {
            "image": image_np,
            "label": label,
            "dataset_idx": metadata["idx"],
            "source": metadata["source"],
        }

    def custom_trajectory_naming(trajectory_group, index):
        """Custom trajectory naming based on content."""
        first_idx = trajectory_group[0]
        last_idx = trajectory_group[-1]
        return f"vision_batch_{first_idx:06d}_to_{last_idx:06d}"

    # Create adapter manually
    pytorch_dataset = CustomVisionDataset(num_samples=1000)

    adapter = PyTorchDatasetAdapter(
        dataset=pytorch_dataset,
        transform_fn=custom_transform,
        group_size=200,  # 200 images per trajectory
        trajectory_name_fn=custom_trajectory_naming,
    )

    # Use the adapter with the ingestion system
    vla_dataset = create_vla_dataset_from_source(
        data_source=adapter,
        output_directory="./manual_adapter_trajectories",
        num_workers=4,
    )

    return vla_dataset


# Example 5: Working with DataLoader
def example_dataloader_integration():
    """Show how to work with PyTorch DataLoader."""

    # Create dataset and dataloader
    pytorch_dataset = CustomVisionDataset(num_samples=1000)
    dataloader = torch.utils.data.DataLoader(pytorch_dataset,
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=2)

    # Convert dataloader to iterator for ingestion
    def dataloader_iterator():
        """Convert DataLoader to iterator of individual items."""
        for batch in dataloader:
            images, labels, metadata_list = batch

            # Yield individual items from the batch
            for i in range(len(images)):
                yield (
                    images[i],
                    labels[i].item(),
                    {
                        k: v[i] if isinstance(v, list) else v
                        for k, v in metadata_list.items()
                    },
                )

    def transform_batch_item(item):
        """Transform individual item from batched data."""
        image, label, metadata = item

        return {
            "image": image.numpy().transpose(1, 2, 0),
            "label": label,
            "metadata": metadata,
        }

    # Create VLA dataset from dataloader
    vla_dataset = create_vla_dataset_from_source(
        data_source=dataloader_iterator,
        transform_fn=transform_batch_item,
        output_directory="./dataloader_trajectories",
        num_workers=4,
        group_size=100,
    )

    return vla_dataset


# Example 6: Handling large datasets with streaming
def example_large_dataset_streaming():
    """Example for very large datasets that don't fit in memory."""

    class LargeDataset(torch.utils.data.Dataset):
        """Simulated large dataset."""

        def __init__(self, num_samples: int = 100000):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Simulate loading from disk/database
            return {
                "data": torch.randn(1000),  # Large data item
                "id": idx,
                "metadata": {
                    "partition": idx // 1000
                },
            }

    large_dataset = LargeDataset(num_samples=10000)

    # Process in smaller groups to manage memory
    vla_dataset = create_vla_dataset_from_source(
        data_source=large_dataset,
        output_directory="./large_dataset_trajectories",
        num_workers=8,  # More workers for parallel processing
        group_size=1000,  # Larger groups for efficiency
        # Additional config for large datasets
        raw_codec="rawvideo_pyarrow",  # Efficient compression
        shuffle_items=True,  # Shuffle for better training
    )

    return vla_dataset


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== PyTorch Integration Examples ===\n")

    # Run examples
    examples = [
        ("Simple conversion", example_simple_pytorch_conversion),
        ("With transform", example_pytorch_with_transform),
        ("Time series", example_timeseries_pytorch),
        ("Manual adapter", example_manual_adapter),
        ("DataLoader integration", example_dataloader_integration),
        ("Large dataset streaming", example_large_dataset_streaming),
    ]

    for name, example_func in examples:
        print(f"Running: {name}")
        try:
            dataset = example_func()
            print(f"  ✓ Success: {dataset.count()} items")

            # Show peek for first few examples
            if name in ["Simple conversion", "With transform"]:
                first_item = dataset.peek()
                if first_item:
                    print(f"  Sample keys: {list(first_item.keys())}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print("All examples completed!")
