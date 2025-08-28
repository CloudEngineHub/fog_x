"""
Executor module for running generated code on Ray datasets.
"""

import logging
from typing import Any, Callable, Dict, List, Union

import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class Executor:
    """
    Executor for running LLM-generated functions on Ray datasets.

    Provides safe execution environment and handles Ray dataset operations
    like filtering, mapping, and aggregation.
    """

    def __init__(self, max_retries: int = 3, tools_manager=None):
        """
        Initialize Executor.

        Args:
            max_retries: Maximum number of retries for failed operations
            tools_manager: ToolsManager instance for accessing tools
        """
        self.max_retries = max_retries
        self.tools_manager = tools_manager

    def apply_filter(self, dataset,
                     filter_func: Callable[[Dict[str, Any]], bool]):
        """
        Apply filter function to Ray dataset or VLADataset.

        Args:
            dataset: Input Ray dataset or VLADataset
            filter_func: Filter function that returns True for trajectories to keep

        Returns:
            Filtered dataset (same type as input)
        """
        # Check if this is a VLADataset or DroidDataset
        if hasattr(dataset, 'filter') and (hasattr(dataset, '_is_loaded') or hasattr(dataset, '_is_downloaded')):
            # Use dataset's built-in filter which handles lazy loading
            dataset_type = type(dataset).__name__
            logger.info(f"Using {dataset_type} filter method")
            return dataset.filter(filter_func)
            
        # Otherwise treat as Ray dataset
        try:
            # Wrap filter function for Ray dataset
            def ray_filter_wrapper(batch):
                """Wrapper to apply filter function to batches."""
                import pandas as pd

                # Convert pandas DataFrame to dict format if needed
                if isinstance(batch, pd.DataFrame):
                    batch_dict = batch.to_dict("list")
                else:
                    batch_dict = batch

                # Convert batch format to individual trajectories
                batch_size = len(next(iter(batch_dict.values())))
                keep_flags = []

                for i in range(batch_size):
                    # Extract single trajectory from batch
                    trajectory = {
                        key: values[i]
                        for key, values in batch_dict.items()
                    }

                    try:
                        # Apply filter function
                        keep = filter_func(trajectory)
                        keep_flags.append(bool(keep))
                    except Exception as e:
                        logger.warning(
                            f"Filter function failed for trajectory {i}: {e}")
                        keep_flags.append(False)

                # Return original data WITH __keep__ column added
                if isinstance(batch, pd.DataFrame):
                    # Add __keep__ column to existing batch
                    batch_with_keep = batch.copy()
                    batch_with_keep["__keep__"] = keep_flags
                    return batch_with_keep
                else:
                    # Add __keep__ column to existing batch_dict (copy to avoid mutation)
                    batch_dict_with_keep = batch_dict.copy()
                    batch_dict_with_keep["__keep__"] = keep_flags
                    return batch_dict_with_keep

            # Apply filter using Ray's map_batches and filter
            filtered_dataset = dataset.map_batches(ray_filter_wrapper,
                                                   batch_format="pandas")
            filtered_dataset = filtered_dataset.filter(
                lambda batch: batch["__keep__"])

            # Remove the temporary __keep__ column
            def remove_keep_column(batch):
                import pandas as pd

                if isinstance(batch, pd.DataFrame):
                    return batch.drop(columns=["__keep__"], errors="ignore")
                else:
                    return {k: v for k, v in batch.items() if k != "__keep__"}

            return filtered_dataset.map_batches(remove_keep_column,
                                                batch_format="pandas")

        except Exception as e:
            logger.error(f"Filter operation failed: {e}")
            raise RuntimeError(f"Failed to apply filter: {e}")

    def apply_map(
            self, dataset,
            map_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Apply map function to Ray dataset or VLADataset.

        Args:
            dataset: Input Ray dataset or VLADataset
            map_func: Map function that transforms trajectories

        Returns:
            Transformed dataset (same type as input)
        """
        # Check if this is a VLADataset or DroidDataset
        if hasattr(dataset, 'map') and (hasattr(dataset, '_is_loaded') or hasattr(dataset, '_is_downloaded')):
            # Use dataset's built-in map which handles lazy loading
            return dataset.map(map_func)
            
        # Otherwise treat as Ray dataset
        try:
            # Wrap map function for Ray dataset
            def ray_map_wrapper(batch):
                """Wrapper to apply map function to batches."""
                import pandas as pd

                # Convert pandas DataFrame to dict format if needed
                if isinstance(batch, pd.DataFrame):
                    batch_dict = batch.to_dict("list")
                else:
                    batch_dict = batch

                batch_size = len(next(iter(batch_dict.values())))
                transformed_batch: Dict[str, List[Any]] = {}

                for i in range(batch_size):
                    # Extract single trajectory from batch
                    trajectory = {
                        key: values[i]
                        for key, values in batch_dict.items()
                    }

                    try:
                        # Apply map function
                        transformed_trajectory = map_func(trajectory)

                        # Ensure the transformed result is a dictionary. If the
                        # map function returns a scalar / list / bool we wrap it
                        # into a dictionary under the generic key "result" so
                        # that downstream operations have a consistent schema.
                        if not isinstance(transformed_trajectory, dict):
                            transformed_trajectory = {"result": transformed_trajectory}

                        # Accumulate results
                        for key, value in transformed_trajectory.items():
                            if key not in transformed_batch:
                                transformed_batch[key] = []
                            transformed_batch[key].append(value)

                    except Exception as e:
                        logger.warning(
                            f"Map function failed for trajectory {i}: {e}")
                        # Keep original trajectory on error
                        for key, value in trajectory.items():
                            if key not in transformed_batch:
                                transformed_batch[key] = []
                            transformed_batch[key].append(value)

                # Return in appropriate format
                if isinstance(batch, pd.DataFrame):
                    return pd.DataFrame(transformed_batch)
                else:
                    return transformed_batch

            # Apply map using Ray's map_batches
            return dataset.map_batches(ray_map_wrapper, batch_format="pandas")

        except Exception as e:
            logger.error(f"Map operation failed: {e}")
            raise RuntimeError(f"Failed to apply map: {e}")

    def apply_aggregation(
            self, dataset: Dataset, agg_func: Callable[[List[Dict[str, Any]]],
                                                       Any]) -> Any:
        """
        Apply aggregation function to Ray dataset.

        Args:
            dataset: Input Ray dataset
            agg_func: Aggregation function that processes list of trajectories

        Returns:
            Aggregation result
        """
        try:
            # Collect all trajectories (for small datasets)
            # For large datasets, consider implementing distributed aggregation
            trajectories = self._collect_trajectories(dataset)

            # Apply aggregation function
            result = agg_func(trajectories)

            return result

        except Exception as e:
            logger.error(f"Aggregation operation failed: {e}")
            raise RuntimeError(f"Failed to apply aggregation: {e}")

    def apply_analysis(
            self, dataset: Dataset,
            analysis_func: Callable[[List[Dict[str, Any]]], str]) -> str:
        """
        Apply analysis function to Ray dataset.

        Args:
            dataset: Input Ray dataset
            analysis_func: Analysis function that returns string description

        Returns:
            Analysis result as string
        """
        try:
            # Collect trajectories for analysis
            trajectories = self._collect_trajectories(dataset)

            # Apply analysis function
            result = analysis_func(trajectories)

            return str(result)

        except Exception as e:
            logger.error(f"Analysis operation failed: {e}")
            raise RuntimeError(f"Failed to apply analysis: {e}")

    def _collect_trajectories(
            self,
            dataset: Dataset,
            max_trajectories: int = 10000) -> List[Dict[str, Any]]:
        """
        Collect trajectories from Ray dataset into list.

        Args:
            dataset: Input Ray dataset
            max_trajectories: Maximum number of trajectories to collect

        Returns:
            List of trajectory dictionaries
        """
        try:
            # Get dataset count
            count = dataset.count()

            # Use take() method instead of to_pandas() to avoid tensor casting issues
            # This is more reliable for datasets with complex numpy arrays
            if count > max_trajectories:
                logger.warning(
                    f"Dataset has {count} trajectories, sampling {max_trajectories}"
                )
                # Sample random trajectories and take them
                sampled_dataset = dataset.random_sample(max_trajectories / count)
                trajectories = sampled_dataset.take(max_trajectories)
            else:
                # Collect all trajectories using take()
                trajectories = dataset.take(count)

            return trajectories

        except Exception as e:
            logger.error(f"Failed to collect trajectories: {e}")
            # Final fallback: try to get a small number of items
            try:
                return dataset.take(min(max_trajectories, 100))
            except:
                raise RuntimeError(f"Failed to collect trajectories: {e}")

    def validate_function(self, func: Callable,
                          expected_signature: str) -> bool:
        """
        Validate that a function has the expected signature.

        Args:
            func: Function to validate
            expected_signature: Expected function signature string

        Returns:
            True if function is valid
        """
        try:
            import inspect

            # Get function signature
            sig = inspect.signature(func)

            # Basic validation - check parameter count and names
            params = list(sig.parameters.keys())

            if "filter" in expected_signature:
                return len(params) == 1 and "trajectory" in params[0]
            elif "map" in expected_signature:
                return len(params) == 1 and "trajectory" in params[0]
            elif "aggregat" in expected_signature:
                return len(params) == 1 and "trajectories" in params[0]
            elif "analys" in expected_signature:
                return len(params) == 1 and "trajectories" in params[0]

            return True

        except Exception as e:
            logger.warning(f"Function validation failed: {e}")
            return False

    def safe_execute(self, func: Callable, *args,
                     **kwargs) -> Union[Any, Exception]:
        """
        Safely execute a function with error handling and retries.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or Exception if all retries failed
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Function execution attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Add small delay before retry
                    import time

                    time.sleep(0.1 * (attempt + 1))

        return last_exception

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        # This could be extended to track execution metrics
        return {
            "max_retries":
            self.max_retries,
            "ray_cluster_resources":
            (ray.cluster_resources() if ray.is_initialized() else {}),
        }

    def __repr__(self) -> str:
        """String representation of Executor."""
        return f"Executor(max_retries={self.max_retries})"
