"""
Agent class for natural language dataset processing with RoboDM Ray datasets.
"""

from typing import Any, Callable, Dict, List, Optional

import ray
from ray.data import Dataset

from .executor import Executor
from .planner import Planner
from .tools import ToolsManager, create_manager


class Agent:
    """
    Agent for processing RoboDM Ray datasets using natural language prompts.

    Provides high-level interface for dataset operations like filtering,
    mapping, and analysis using LLM-generated code.
    """

    def __init__(
        self,
        dataset,
        llm_model: str = "Llama 3.2-Vision2.5-7b",
        tools_config: Optional[Dict[str, Any]] = None,
        **llm_kwargs
    ):
        """
        Initialize Agent with a RoboDM dataset.

        Args:
            dataset: Ray Dataset or VLADataset containing trajectory data
            llm_model: Model name for LLM-based planning (default: Llama 3.2-Vision2.5-7b)
            tools_config: Configuration for tools system (can be dict or preset name)
            **llm_kwargs: Additional LLM configuration (e.g., context_length, enforce_eager)
        """
        self.dataset = dataset

        # Handle tools configuration
        if isinstance(tools_config, str):
            # It's a preset name
            self.tools_manager = create_manager(tools_config)
        else:
            # It's a configuration dict or None
            self.tools_manager = ToolsManager(config=tools_config)

        # Pass LLM configuration to Planner
        self.planner = Planner(llm_model=llm_model,
                               tools_manager=self.tools_manager,
                               **llm_kwargs)
        self.executor = Executor(tools_manager=self.tools_manager)

    def filter(self, prompt: str) -> Dataset:
        """
        Filter trajectories using natural language prompt.

        Args:
            prompt: Natural language description of filter criteria
                   e.g., "trajectories that have occluded views"

        Returns:
            Filtered Ray Dataset

        Example:
            >>> agent = Agent(robodm_dataset)
            >>> filtered = agent.filter("trajectories that have occluded views")
        """
        # Generate filter function using planner with dataset schema
        filter_func = self.planner.generate_filter_function(
            prompt, dataset=self.dataset)

        # Execute filter function on dataset
        return self.executor.apply_filter(self.dataset, filter_func)

    def map(self, prompt: str) -> Dataset:
        """
        Transform trajectories using natural language prompt.

        Args:
            prompt: Natural language description of transformation
                   e.g., "add frame difference features"

        Returns:
            Transformed Ray Dataset
        """
        # Generate map function using planner with dataset schema
        map_func = self.planner.generate_map_function(prompt,
                                                      dataset=self.dataset)

        # Execute map function on dataset
        return self.executor.apply_map(self.dataset, map_func)

    def aggregate(self, prompt: str) -> Any:
        """
        Aggregate dataset using natural language prompt.

        Args:
            prompt: Natural language description of aggregation
                   e.g., "count trajectories by scene type"

        Returns:
            Aggregation result
        """
        # Generate aggregation function using planner with dataset schema
        agg_func = self.planner.generate_aggregation_function(
            prompt, dataset=self.dataset)

        # Execute aggregation function on dataset
        return self.executor.apply_aggregation(self.dataset, agg_func)

    def analyze(self, prompt: str) -> str:
        """
        Analyze dataset using natural language prompt.

        Args:
            prompt: Natural language description of analysis
                   e.g., "what is the average trajectory length?"

        Returns:
            Analysis result as string
        """
        # Generate analysis function using planner with dataset schema
        analysis_func = self.planner.generate_analysis_function(
            prompt, dataset=self.dataset)

        # Execute analysis function on dataset
        return self.executor.apply_analysis(self.dataset, analysis_func)

    def count(self) -> int:
        """Get count of trajectories in dataset."""
        return self.dataset.count()

    def take(self, n: int = 10) -> list:
        """Take first n trajectories from dataset."""
        return self.dataset.take(n)

    def schema(self) -> Dict[str, Any]:
        """Get schema information of the dataset."""
        try:
            # Try Ray dataset schema first
            return self.dataset.schema()
        except:
            # Fallback to planner's schema inspection
            return self.planner.inspect_dataset_schema(self.dataset)

    def inspect_schema(self) -> Dict[str, Any]:
        """Get detailed schema inspection including shapes, types, and semantic information."""
        return self.planner.inspect_dataset_schema(self.dataset)

    def describe_dataset(self) -> str:
        """Get a human-readable description of the dataset structure."""
        schema_info = self.inspect_schema()

        if not schema_info["keys"]:
            return "Empty dataset or unable to inspect schema."

        description = f"Dataset with {len(schema_info['keys'])} feature keys:\n"

        for key in schema_info["keys"]:
            if key in schema_info["shapes"]:
                shape = schema_info["shapes"][key]
                dtype = schema_info["dtypes"].get(key, "unknown")
                description += f"  • {key}: {dtype} array, shape {shape}"

                if key in schema_info["image_keys"]:
                    description += " (image data)\n"
                elif key in schema_info["temporal_keys"]:
                    description += " (temporal sequence)\n"
                else:
                    description += "\n"
            else:
                sample_val = schema_info["sample_values"].get(key, "...")
                description += (
                    f"  • {key}: {type(sample_val).__name__} = {sample_val}\n")

        return description.strip()

    def configure_tools(self, config: Dict[str, Any]):
        """Configure tools system."""
        self.tools_manager.update_config(config)

    def list_tools(self) -> List[str]:
        """List available tools."""
        return self.tools_manager.list_tools()

    def enable_tool(self, tool_name: str):
        """Enable a specific tool."""
        self.tools_manager.enable_tool(tool_name)

    def disable_tool(self, tool_name: str):
        """Disable a specific tool."""
        self.tools_manager.disable_tool(tool_name)

    def get_tools_info(self) -> str:
        """Get information about available tools."""
        return self.tools_manager.get_tools_prompt()

    def __len__(self) -> int:
        """Get count of trajectories in dataset."""
        return self.count()

    def __repr__(self) -> str:
        """String representation of Agent."""
        return f"Agent(dataset={self.dataset}, count={len(self)})"
