"""
Planner module for generating code using LLM based on natural language prompts.
"""

import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from vllm import LLM, SamplingParams
except ImportError:
    # Fallback for when vllm is not installed
    class LLM:

        def __init__(self, model: str):
            self.model = model

        def generate(self, prompts, sampling_params):
            # Mock response
            class MockOutput:

                def __init__(self):
                    self.outputs = [MockGeneration()]

            class MockGeneration:

                def __init__(self):
                    self.text = "# Mock LLM response - vllm not installed\nreturn True"

            return [MockOutput()]

    class SamplingParams:

        def __init__(self, **kwargs):
            self.params = kwargs


class Planner:
    """
    LLM-based planner that generates Python code for dataset operations.

    Takes natural language prompts and generates executable functions
    for filtering, mapping, and analyzing robotic trajectory data.
    Dynamically adapts to dataset schema.
    """

    def __init__(self, llm_model: str = "Llama 3.2-Vision", tools_manager=None):
        """
        Initialize Planner with specified LLM model.

        Args:
            llm_model: Model name for code generation (default: Llama 3.2-Vision)
            tools_manager: ToolsManager instance for accessing tools
        """
        self.llm_model = llm_model
        self.llm = LLM(model=llm_model)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            stop=["```", "# End of function"],
        )
        self.tools_manager = tools_manager
        self._cached_schema = None
        self._cached_sample = None

    def inspect_dataset_schema(self, dataset) -> Dict[str, Any]:
        """
        Inspect dataset schema and cache the result.

        Args:
            dataset: Ray dataset to inspect

        Returns:
            Dictionary with schema information
        """
        if self._cached_schema is not None:
            return self._cached_schema

        try:
            # Get sample data to understand structure
            sample_data = dataset.take(1)[0] if dataset.count() > 0 else {}

            # Analyze the schema
            schema_info = {
                "keys": list(sample_data.keys()),
                "shapes": {},
                "dtypes": {},
                "sample_values": {},
                "has_images": False,
                "image_keys": [],
                "temporal_keys": [],
                "scalar_keys": [],
            }

            for key, value in sample_data.items():
                if hasattr(value, "shape"):
                    schema_info["shapes"][key] = list(value.shape)
                    schema_info["dtypes"][key] = str(value.dtype)

                    # Check if this looks like image data
                    if len(value.shape) >= 3 and value.shape[-1] in [
                            1,
                            3,
                            4,
                    ]:  # H,W,C format
                        schema_info["has_images"] = True
                        schema_info["image_keys"].append(key)

                    # Check if this looks like temporal data (first dim > 1)
                    if len(value.shape) >= 2 and value.shape[0] > 1:
                        schema_info["temporal_keys"].append(key)

                    # Store a sample for reference
                    if isinstance(value, np.ndarray) and value.size < 10:
                        schema_info["sample_values"][key] = value.tolist()
                else:
                    # Scalar or other types
                    schema_info["scalar_keys"].append(key)
                    schema_info["sample_values"][key] = value

            self._cached_schema = schema_info
            return schema_info

        except Exception as e:
            # Fallback schema
            return {
                "keys": [],
                "shapes": {},
                "dtypes": {},
                "sample_values": {},
                "has_images": False,
                "image_keys": [],
                "temporal_keys": [],
                "scalar_keys": [],
                "error": str(e),
            }

    def _generate_schema_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Generate schema description for LLM prompt."""
        if not schema_info["keys"]:
            return "# Unknown schema - use trajectory.keys() to explore"

        schema_desc = "# Dataset Schema:\n"

        for key in schema_info["keys"]:
            if key in schema_info["shapes"]:
                shape = schema_info["shapes"][key]
                dtype = schema_info["dtypes"].get(key, "unknown")
                schema_desc += (
                    f"# trajectory['{key}'] -> {dtype} array, shape {shape}\n")

                # Add semantic hints
                if key in schema_info["image_keys"]:
                    schema_desc += f"#   -> Image data (use robo2vlm for analysis)\n"
                elif key in schema_info["temporal_keys"]:
                    schema_desc += f"#   -> Temporal sequence data\n"
            else:
                sample_val = schema_info["sample_values"].get(key, "...")
                schema_desc += f"# trajectory['{key}'] -> {type(sample_val).__name__}: {sample_val}\n"

        return schema_desc

    def generate_filter_function(
            self,
            prompt: str,
            dataset=None) -> Callable[[Dict[str, Any]], bool]:
        """
        Generate a filter function based on natural language prompt.

        Args:
            prompt: Natural language description of filter criteria
            dataset: Dataset to inspect for schema (optional)

        Returns:
            Function with signature: def filter_func(trajectory: Dict[str, Any]) -> bool
        """
        # Get schema information if dataset provided
        schema_info = {}
        schema_prompt = ""
        if dataset is not None:
            schema_info = self.inspect_dataset_schema(dataset)
            schema_prompt = self._generate_schema_prompt(schema_info)

        # Get tools information
        tools_prompt = ""
        if self.tools_manager is not None:
            tools_prompt = self.tools_manager.get_tools_prompt()

        system_prompt = f"""You are a Python code generator for robotic trajectory filtering.
Generate ONLY the function body for a filter function with this exact signature:
def has_condition(trajectory: Dict[str, Any]) -> bool:

{tools_prompt}

{schema_prompt}

Return only the function body (no imports, no function definition line).
Use the actual dataset schema above to access the correct trajectory keys.
Use the available tools for analysis operations.

Example patterns:
- For image analysis: robo2vlm(frame, "question about image")
- For image properties: analyze_image(frame, "blur")
- For trajectory analysis: analyze_trajectory(data, "statistics")
- For array operations: np.mean(trajectory["key_name"])
- For temporal analysis: len(trajectory["temporal_key"])
- For metadata: trajectory.get("metadata", {{}}).get("field")"""

        full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nFunction body:"

        outputs = self.llm.generate([full_prompt], self.sampling_params)
        generated_code = outputs[0].outputs[0].text.strip()

        # Clean up generated code
        function_body = self._clean_generated_code(generated_code)

        # Create complete function
        complete_function = f"""def has_condition(trajectory: Dict[str, Any]) -> bool:
{function_body}"""

        # Compile and return function
        return self._compile_function(complete_function, "has_condition")

    def generate_map_function(
            self,
            prompt: str,
            dataset=None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a map function based on natural language prompt.

        Args:
            prompt: Natural language description of transformation
            dataset: Dataset to inspect for schema (optional)

        Returns:
            Function with signature: def map_func(trajectory: Dict[str, Any]) -> Dict[str, Any]
        """
        # Get schema information if dataset provided
        schema_info = {}
        schema_prompt = ""
        if dataset is not None:
            schema_info = self.inspect_dataset_schema(dataset)
            schema_prompt = self._generate_schema_prompt(schema_info)

        # Get tools information
        tools_prompt = ""
        if self.tools_manager is not None:
            tools_prompt = self.tools_manager.get_tools_prompt()

        system_prompt = f"""You are a Python code generator for robotic trajectory transformation.
Generate ONLY the function body for a map function with this exact signature:
def transform_trajectory(trajectory: Dict[str, Any]) -> Dict[str, Any]:

{tools_prompt}

{schema_prompt}

Return only the function body (no imports, no function definition line).
Use the actual dataset schema above to access the correct trajectory keys.
Use the available tools for analysis and processing operations.
You must return a modified copy of the trajectory dictionary.

Example patterns:
- result = trajectory.copy()  # Always start with a copy
- For image processing: new_images = process_images(trajectory["image_key"])
- For feature engineering: new_feature = compute_feature(trajectory["data_key"])
- For tool usage: blur_info = analyze_image(frame, "blur")
- result["new_key"] = new_feature  # Add new features
- return result  # Always return the modified trajectory"""

        full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nFunction body:"

        outputs = self.llm.generate([full_prompt], self.sampling_params)
        generated_code = outputs[0].outputs[0].text.strip()

        # Clean up generated code
        function_body = self._clean_generated_code(generated_code)

        # Create complete function
        complete_function = f"""def transform_trajectory(trajectory: Dict[str, Any]) -> Dict[str, Any]:
{function_body}"""

        # Compile and return function
        return self._compile_function(complete_function,
                                      "transform_trajectory")

    def generate_aggregation_function(self,
                                      prompt: str,
                                      dataset=None) -> Callable[[list], Any]:
        """
        Generate an aggregation function based on natural language prompt.

        Args:
            prompt: Natural language description of aggregation
            dataset: Dataset to inspect for schema (optional)

        Returns:
            Function with signature: def agg_func(trajectories: list) -> Any
        """
        # Get schema information if dataset provided
        schema_info = {}
        schema_prompt = ""
        if dataset is not None:
            schema_info = self.inspect_dataset_schema(dataset)
            schema_prompt = self._generate_schema_prompt(schema_info).replace(
                "trajectory[", "traj[")

        system_prompt = f"""You are a Python code generator for robotic trajectory aggregation.
Generate ONLY the function body for an aggregation function with this exact signature:
def aggregate_trajectories(trajectories: list) -> Any:

Available tools:
- robo2vlm(frame, prompt): Vision-language model for image analysis
- trajectories is a list of Dict[str, Any] containing trajectory data
- Use efficient numpy/pandas operations for large datasets

{schema_prompt}

Return only the function body (no imports, no function definition line).
Use the actual dataset schema above to access the correct trajectory keys (replace 'trajectory[' with 'traj[').

Example patterns:
- for traj in trajectories: ...  # Iterate through trajectories
- Use traj["key_name"] to access trajectory data
- For statistics: lengths = [len(traj["temporal_key"]) for traj in trajectories]
- For grouping: group_by_field = defaultdict(list)"""

        full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nFunction body:"

        outputs = self.llm.generate([full_prompt], self.sampling_params)
        generated_code = outputs[0].outputs[0].text.strip()

        # Clean up generated code
        function_body = self._clean_generated_code(generated_code)

        # Create complete function
        complete_function = f"""def aggregate_trajectories(trajectories: list) -> Any:
{function_body}"""

        # Compile and return function
        return self._compile_function(complete_function,
                                      "aggregate_trajectories")

    def generate_analysis_function(self,
                                   prompt: str,
                                   dataset=None) -> Callable[[list], str]:
        """
        Generate an analysis function based on natural language prompt.

        Args:
            prompt: Natural language description of analysis
            dataset: Dataset to inspect for schema (optional)

        Returns:
            Function with signature: def analysis_func(trajectories: list) -> str
        """
        # Get schema information if dataset provided
        schema_info = {}
        schema_prompt = ""
        if dataset is not None:
            schema_info = self.inspect_dataset_schema(dataset)
            schema_prompt = self._generate_schema_prompt(schema_info).replace(
                "trajectory[", "traj[")

        system_prompt = f"""You are a Python code generator for robotic trajectory analysis.
Generate ONLY the function body for an analysis function with this exact signature:
def analyze_trajectories(trajectories: list) -> str:

Available tools:
- robo2vlm(frame, prompt): Vision-language model for image analysis
- trajectories is a list of Dict[str, Any] containing trajectory data
- Return a descriptive string with analysis results

{schema_prompt}

Return only the function body (no imports, no function definition line).
Use the actual dataset schema above to access the correct trajectory keys (replace 'trajectory[' with 'traj[').

Example patterns:
- for traj in trajectories: ...  # Iterate through trajectories
- Use traj["key_name"] to access trajectory data
- Calculate statistics and return formatted string
- return f"Analysis result: {value:.2f}" """

        full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nFunction body:"

        outputs = self.llm.generate([full_prompt], self.sampling_params)
        generated_code = outputs[0].outputs[0].text.strip()

        # Clean up generated code
        function_body = self._clean_generated_code(generated_code)

        # Create complete function
        complete_function = f"""def analyze_trajectories(trajectories: list) -> str:
{function_body}"""

        # Compile and return function
        return self._compile_function(complete_function,
                                      "analyze_trajectories")

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by adding proper indentation."""
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.strip():
                # Add 4-space indentation if not already indented
                if not line.startswith("    ") and not line.startswith("\t"):
                    cleaned_lines.append("    " + line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append("")

        return "\n".join(cleaned_lines)

    def _compile_function(self, function_code: str,
                          function_name: str) -> Callable:
        """Compile generated function code and return callable."""
        # Create execution environment with necessary imports and tools
        exec_globals = {
            "Dict": Dict,
            "Any": Any,
            "np": np,
            "__builtins__": __builtins__,
        }

        # Add tools to execution environment
        if self.tools_manager is not None:
            tools_namespace = self.tools_manager.get_tools_namespace()
            exec_globals.update(tools_namespace)

        try:
            # Execute the function definition
            exec(function_code, exec_globals)

            # Return the compiled function
            return exec_globals[function_name]

        except Exception as e:
            raise RuntimeError(
                f"Failed to compile generated function: {e}\nGenerated code:\n{function_code}"
            )

    def __repr__(self) -> str:
        """String representation of Planner."""
        return f"Planner(model={self.llm_model})"
