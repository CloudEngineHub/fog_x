"""
VLM-Based Robot Demonstration Quality Scoring

This script evaluates the quality of robot demonstrations using Vision-Language Models
to score various factors like visual clarity, occlusion, scene complexity, etc.
The scoring system is modular and easily adjustable.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import numpy as np
import cv2
import ray
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod

from robodm.dataset import VLADataset, DatasetConfig
from robodm.agent.vlm_service import get_vlm_service


@dataclass
class ScoringConfig:
    """Configuration for the scoring system."""
    # Weights for each scoring component
    weights: Dict[str, float] = None
    
    # Thresholds for quality levels
    thresholds: Dict[str, float] = None
    
    # Number of frames to sample per trajectory
    frames_per_trajectory: int = 6
    
    # Number of VLM queries per scoring component (for averaging)
    vlm_queries_per_score: int = 3
    
    # Whether to save all images or only top N
    save_all_images: bool = True
    top_n_images: int = 1000000
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "visual_clarity": 0.35,
                "occlusion": 0.25,
                "scene_complexity": 0.15,
                "task_atomicity": 0.15,
                "target_object_quality": 0.10
            }
        
        if self.thresholds is None:
            self.thresholds = {
                "excellent": 0.8,
                "good": 0.6,
                "fair": 0.4,
                "poor": 0.2
            }


class QualityScorer(ABC):
    """Abstract base class for quality scoring modules."""
    
    @abstractmethod
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        """
        Score the quality aspect.
        
        Returns:
            Tuple of (score between 0-1, one-sentence explanation, full VLM response)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this scorer."""
        pass


class VisualClarityScorer(QualityScorer):
    """Scores visual clarity including lighting, focus, and contrast."""
    
    def get_name(self) -> str:
        return "visual_clarity"
    
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        # Create a grid of frames for better context
        if len(frames) >= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4])
            combined_frame = np.vstack([top_row, bottom_row])
        elif len(frames) >= 2:
            combined_frame = np.hstack(frames)
        else:
            combined_frame = frames[0]
        
        task_context = f"\nThe robot is performing the task: '{language_instruction}'" if language_instruction else ""
        
        prompt = f"""Looking at this robot manipulation sequence, rate the visual quality on a scale of 0-100.{task_context}
Consider lighting, focus, and contrast for evaluating how well the robot task can be observed.
Provide ONLY:
1. A single score (0-100)
2. One sentence explanation

Format: Score: [number]. [One sentence explanation]"""

        # Query VLM multiple times and average results
        scores = []
        explanations = []
        all_responses = []
        
        for i in range(num_queries):
            response = vlm_service.analyze_image(combined_frame, prompt)
            all_responses.append(response)
            
            try:
                import re
                # Find first number that could be a score
                numbers = re.findall(r'\b(\d{1,3})\b', response)
                valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                
                if valid_scores:
                    query_score = valid_scores[0] / 100.0
                else:
                    query_score = 0.7  # Default
                
                # Extract one sentence explanation
                sentences = response.split('.')
                query_explanation = sentences[1].strip() if len(sentences) > 1 else "Visual quality assessed."
                
                scores.append(query_score)
                explanations.append(query_explanation)
                
            except Exception as e:
                scores.append(0.7)
                explanations.append("Failed to parse VLM response.")
        
        # Average the scores and use the first explanation
        final_score = sum(scores) / len(scores) if scores else 0.7
        final_explanation = explanations[0] if explanations else "Visual quality assessed."
        combined_response = "\n---\n".join(all_responses)
        
        return final_score, final_explanation, combined_response


class OcclusionScorer(QualityScorer):
    """Scores occlusion of target objects and robot gripper."""
    
    def get_name(self) -> str:
        return "occlusion"
    
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        # Create combined frame
        if len(frames) >= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4])
            combined_frame = np.vstack([top_row, bottom_row])
        elif len(frames) >= 2:
            combined_frame = np.hstack(frames)
        else:
            combined_frame = frames[0]
        
        task_context = f"\nThe robot is performing the task: '{language_instruction}'" if language_instruction else ""
        
        prompt = f"""Rate the visibility/occlusion in this robot manipulation sequence on a scale of 0-100.{task_context}
100 = Perfect visibility, no occlusion of important objects/gripper for this task
0 = Severe occlusion, can't see key objects/gripper needed for this task
Provide ONLY:
1. A single score (0-100)
2. One sentence explanation

Format: Score: [number]. [One sentence explanation]"""

        # Query VLM multiple times and average results
        scores = []
        explanations = []
        all_responses = []
        
        for i in range(num_queries):
            response = vlm_service.analyze_image(combined_frame, prompt)
            all_responses.append(response)
            
            try:
                import re
                numbers = re.findall(r'\b(\d{1,3})\b', response)
                valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                
                if valid_scores:
                    query_score = valid_scores[0] / 100.0
                else:
                    query_score = 0.8
                
                sentences = response.split('.')
                query_explanation = sentences[1].strip() if len(sentences) > 1 else "Occlusion level assessed."
                
                scores.append(query_score)
                explanations.append(query_explanation)
                
            except Exception as e:
                scores.append(0.8)
                explanations.append("Failed to parse VLM response.")
        
        # Average the scores and use the first explanation
        final_score = sum(scores) / len(scores) if scores else 0.8
        final_explanation = explanations[0] if explanations else "Occlusion level assessed."
        combined_response = "\n---\n".join(all_responses)
        
        return final_score, final_explanation, combined_response


class SceneComplexityScorer(QualityScorer):
    """Scores scene complexity and clutter."""
    
    def get_name(self) -> str:
        return "scene_complexity"
    
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        # Create combined frame
        if len(frames) >= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4])
            combined_frame = np.vstack([top_row, bottom_row])
        elif len(frames) >= 2:
            combined_frame = np.hstack(frames)
        else:
            combined_frame = frames[0]
        
        task_context = f"\nThe robot is performing the task: '{language_instruction}'" if language_instruction else ""
        
        prompt = f"""Rate the scene simplicity for manipulation on a scale of 0-100.{task_context}
100 = Very simple scene appropriate for this task (clear workspace, minimal distractions)
0 = Very complex scene that makes this task difficult (many objects, cluttered)
Provide ONLY:
1. A single score (0-100)
2. One sentence explanation

Format: Score: [number]. [One sentence explanation]"""

        # Query VLM multiple times and average results
        scores = []
        explanations = []
        all_responses = []
        
        for i in range(num_queries):
            response = vlm_service.analyze_image(combined_frame, prompt)
            all_responses.append(response)
            
            try:
                import re
                numbers = re.findall(r'\b(\d{1,3})\b', response)
                valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                
                if valid_scores:
                    query_score = valid_scores[0] / 100.0
                else:
                    query_score = 0.7
                
                sentences = response.split('.')
                query_explanation = sentences[1].strip() if len(sentences) > 1 else "Scene complexity assessed."
                
                scores.append(query_score)
                explanations.append(query_explanation)
                
            except Exception as e:
                scores.append(0.7)
                explanations.append("Failed to parse VLM response.")
        
        # Average the scores and use the first explanation
        final_score = sum(scores) / len(scores) if scores else 0.7
        final_explanation = explanations[0] if explanations else "Scene complexity assessed."
        combined_response = "\n---\n".join(all_responses)
        
        return final_score, final_explanation, combined_response


class TaskAtomicityScorer(QualityScorer):
    """Scores whether the task is atomic or composite."""
    
    def get_name(self) -> str:
        return "task_atomicity"
    
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        # Create a grid of all frames for temporal analysis
        if len(frames) >= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4])
            combined_frame = np.vstack([top_row, bottom_row])
        elif len(frames) >= 2:
            combined_frame = np.hstack(frames)
        else:
            combined_frame = frames[0]
        
        task_context = f"\nThe robot should be performing: '{language_instruction}'" if language_instruction else ""
        
        prompt = f"""Count distinct atomic actions in this robot sequence (e.g., pick, place, push).{task_context}
Rate atomicity on scale 0-100:
100 = Single atomic action that matches the expected task
50 = Two actions
33 = Three actions
etc.
Provide ONLY:
1. A single score (0-100)
2. One sentence explanation

Format: Score: [number]. [One sentence explanation]"""

        # Query VLM multiple times and average results
        scores = []
        explanations = []
        all_responses = []
        
        for i in range(num_queries):
            response = vlm_service.analyze_image(combined_frame, prompt)
            all_responses.append(response)
            
            try:
                import re
                numbers = re.findall(r'\b(\d{1,3})\b', response)
                valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                
                if valid_scores:
                    query_score = valid_scores[0] / 100.0
                else:
                    query_score = 0.7
                
                sentences = response.split('.')
                query_explanation = sentences[1].strip() if len(sentences) > 1 else "Task atomicity assessed."
                
                scores.append(query_score)
                explanations.append(query_explanation)
                
            except Exception as e:
                scores.append(0.7)
                explanations.append("Failed to parse VLM response.")
        
        # Average the scores and use the first explanation
        final_score = sum(scores) / len(scores) if scores else 0.7
        final_explanation = explanations[0] if explanations else "Task atomicity assessed."
        combined_response = "\n---\n".join(all_responses)
        
        return final_score, final_explanation, combined_response



class TargetObjectQualityScorer(QualityScorer):
    """Scores the visual quality of target objects."""
    
    def get_name(self) -> str:
        return "target_object_quality"
    
    def score(self, frames: List[np.ndarray], trajectory: Dict[str, Any], vlm_service: Any, num_queries: int = 1, language_instruction: str = "") -> Tuple[float, str, str]:
        # Create combined frame
        if len(frames) >= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4])
            combined_frame = np.vstack([top_row, bottom_row])
        elif len(frames) >= 2:
            combined_frame = np.hstack(frames)
        else:
            combined_frame = frames[0]
        
        task_context = f"\nThe robot is working with objects for the task: '{language_instruction}'" if language_instruction else ""
        
        prompt = f"""Rate the visual quality of the manipulated object(s) on scale 0-100.{task_context}
100 = Perfect visibility and clear details of the target objects for this task
0 = Poor visibility of target objects, hard to identify what the robot is manipulating
Provide ONLY:
1. A single score (0-100)
2. One sentence explanation

Format: Score: [number]. [One sentence explanation]"""

        # Query VLM multiple times and average results
        scores = []
        explanations = []
        all_responses = []
        
        for i in range(num_queries):
            response = vlm_service.analyze_image(combined_frame, prompt)
            all_responses.append(response)
            
            try:
                import re
                numbers = re.findall(r'\b(\d{1,3})\b', response)
                valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                
                if valid_scores:
                    query_score = valid_scores[0] / 100.0
                else:
                    query_score = 0.7
                
                sentences = response.split('.')
                query_explanation = sentences[1].strip() if len(sentences) > 1 else "Object quality assessed."
                
                scores.append(query_score)
                explanations.append(query_explanation)
                
            except Exception as e:
                scores.append(0.7)
                explanations.append("Failed to parse VLM response.")
        
        # Average the scores and use the first explanation
        final_score = sum(scores) / len(scores) if scores else 0.7
        final_explanation = explanations[0] if explanations else "Object quality assessed."
        combined_response = "\n---\n".join(all_responses)
        
        return final_score, final_explanation, combined_response


class TrajectoryQualityBenchmark:
    """Main benchmark class for trajectory quality scoring."""
    
    def __init__(self, 
                 dataset_path: str, 
                 output_dir: str = "./quality_scoring_results",
                 config: Optional[ScoringConfig] = None,
                 scorers: Optional[List[QualityScorer]] = None):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scoring_config = config or ScoringConfig()
        
        # Initialize scorers (no calibration)
        if scorers is None:
            self.scorers = [
                VisualClarityScorer(),
                OcclusionScorer(),
                SceneComplexityScorer(),
                TaskAtomicityScorer(),
                TargetObjectQualityScorer()
            ]
        else:
            self.scorers = scorers
        
        # Dataset configuration
        self.dataset_config = DatasetConfig(
            batch_size=4,
            shuffle=False,
            use_metadata=True,
            auto_build_metadata=False
        )
    
    def load_dataset(self, max_trajectories: Optional[int] = None) -> VLADataset:
        """Load the VLA dataset."""
        print(f"Loading dataset from: {self.dataset_path}")
        
        dataset = VLADataset(
            path=self.dataset_path,
            return_type="numpy",
            config=self.dataset_config
        )
        
        total_trajectories = dataset.count()
        print(f"Found {total_trajectories} trajectory files")
        
        if max_trajectories is not None and total_trajectories > max_trajectories:
            print(f"Limiting to {max_trajectories} trajectories")
            limited_items = dataset.take(max_trajectories)
            
            if limited_items:
                limited_file_paths = [item if isinstance(item, str) else item.get("item", str(item)) 
                                    for item in limited_items]
                
                import ray.data as rd
                limited_ray_dataset = rd.from_items(limited_file_paths)
                
                limited_dataset = VLADataset.__new__(VLADataset)
                limited_dataset.path = dataset.path
                limited_dataset.return_type = dataset.return_type
                limited_dataset.config = dataset.config
                limited_dataset.file_paths = limited_file_paths
                limited_dataset.ray_dataset = limited_ray_dataset
                limited_dataset.metadata_manager = dataset.metadata_manager
                limited_dataset._schema = None
                limited_dataset._stats = None
                limited_dataset._is_loaded = False
                limited_dataset._has_file_paths = True
                
                dataset = limited_dataset
        
        return dataset
    
    def extract_language_instruction(self, trajectory: Dict[str, Any]) -> str:
        """Extract language instruction from trajectory data."""
        # Extract ground truth description
        ground_truth = ""
        current_task = None
        
        # First, check if we have metadata and if it contains raw_data_path
        if 'metadata' in trajectory:
            metadata = trajectory['metadata']
            if hasattr(metadata, '__len__') and len(metadata) > 0:
                metadata_val = metadata[0]
                if isinstance(metadata_val, str):
                    try:
                        import json
                        import os
                        import glob
                        decoded_metadata = json.loads(metadata_val)
                        raw_data_path = decoded_metadata.get('raw_data_path', '')
                        
                        # Try to load the raw metadata JSON file to get current_task
                        if raw_data_path:
                            metadata_pattern = os.path.join(raw_data_path, 'metadata_*.json')
                            metadata_files = glob.glob(metadata_pattern)
                            
                            if metadata_files:
                                with open(metadata_files[0], 'r') as f:
                                    raw_metadata = json.load(f)
                                    current_task = raw_metadata.get('current_task', '')
                                    if current_task:
                                        trajectory["raw_metadata/current_task"] = current_task
                    except Exception as e:
                        pass  # Continue with other methods
        
        # Look for language instruction keys directly in the trajectory
        key_candidates = [
            "tfds/language_instruction",
            "tfds/language_instruction_2", 
            "tfds/language_instruction_3",
            "raw_metadata/current_task"
        ]
        
        found_instructions = []
        
        for key in key_candidates:
            if key == "raw_metadata/current_task":
                if current_task:
                    found_instructions.append(current_task)
            else:
                value = trajectory.get(key, "")
                
                # Check if value exists and has content
                has_content = False
                value_str = ""
                
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        # Handle byte strings
                        val = value[0]
                        if isinstance(val, bytes):
                            value_str = val.decode('utf-8')
                        else:
                            value_str = str(val)
                        has_content = bool(value_str.strip())
                elif isinstance(value, str):
                    value_str = value
                    has_content = bool(value_str.strip())
                elif value:  # For other types
                    value_str = str(value)
                    has_content = bool(value_str.strip())
                
                if has_content:
                    found_instructions.append(value_str)
        
        # Combine all found instructions into ground truth
        if found_instructions:
            ground_truth = "; ".join(found_instructions)
        
        return ground_truth

    def process_single_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single trajectory and compute quality scores."""
        file_path = trajectory.get("__file_path__", "")
        traj_name = Path(file_path).stem
        
        print(f"\n🎯 Processing {traj_name}")
        
        # Extract language instruction
        language_instruction = self.extract_language_instruction(trajectory)
        if language_instruction:
            print(f"  Language instruction: {language_instruction}")
        
        # Initialize results
        results = {
            "trajectory_name": traj_name,
            "file_path": file_path,
            "language_instruction": language_instruction,
            "scores": {},
            "overall_score": 0.0,
            "quality_level": "",
            "explanations": {},
            "frames_saved": []
        }
        
        # Find exterior camera images
        camera_key = None
        for key in trajectory.keys():
            if "raw/images/exterior_image_1" in key:
                camera_key = key
                break
            elif "exterior_image_1" in key and "images" in key:
                camera_key = key
                break
        
        if not camera_key:
            print(f"⚠️  No exterior camera found for {traj_name}")
            return results
        
        images = trajectory.get(camera_key, [])
        if len(images) < self.scoring_config.frames_per_trajectory:
            print(f"⚠️  Not enough frames in {traj_name}")
            return results
        
        # Sample frames evenly
        num_frames = self.scoring_config.frames_per_trajectory
        indices = np.linspace(0, len(images)-1, num_frames, dtype=int)
        selected_frames = [images[i] for i in indices]
        
        # Initialize VLM service
        try:
            vlm_service = get_vlm_service()
            vlm_service.initialize()
        except Exception as e:
            print(f"Error initializing VLM service: {e}")
            return results
        
        # Run each scorer and collect VLM outputs
        vlm_outputs = {}
        num_queries = self.scoring_config.vlm_queries_per_score
        for scorer in self.scorers:
            try:
                score, explanation, full_response = scorer.score(selected_frames, trajectory, vlm_service, num_queries, language_instruction)
                scorer_name = scorer.get_name()
                results["scores"][scorer_name] = score
                results["explanations"][scorer_name] = explanation
                vlm_outputs[scorer_name] = full_response
                print(f"  {scorer_name}: {score:.3f} - {explanation} (avg of {num_queries} queries)")
            except Exception as e:
                print(f"  Error in {scorer.get_name()}: {e}")
                results["scores"][scorer.get_name()] = 0.0
                results["explanations"][scorer.get_name()] = f"Error: {str(e)}"
                vlm_outputs[scorer.get_name()] = f"Error: {str(e)}"
        
        # Calculate overall score
        overall_score = 0.0
        for scorer_name, score in results["scores"].items():
            weight = self.scoring_config.weights.get(scorer_name, 0.0)
            overall_score += score * weight
        
        results["overall_score"] = overall_score
        
        # Determine quality level
        for level, threshold in sorted(self.scoring_config.thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                results["quality_level"] = level
                break
        
        print(f"  Overall Score: {overall_score:.3f} ({results['quality_level']})")
        
        # Save frames - always save unless score is 0
        if overall_score > 0:
            try:
                # Create visualization with all frames
                if len(selected_frames) >= 4:
                    top_row = np.hstack(selected_frames[:2])
                    bottom_row = np.hstack(selected_frames[2:4])
                    combined_frame = np.vstack([top_row, bottom_row])
                elif len(selected_frames) >= 2:
                    combined_frame = np.hstack(selected_frames)
                else:
                    combined_frame = selected_frames[0]
                
                # Ensure the frame is in the right format
                if combined_frame.dtype != np.uint8:
                    if combined_frame.max() <= 1.0:
                        combined_frame = (combined_frame * 255).astype(np.uint8)
                    else:
                        combined_frame = combined_frame.astype(np.uint8)
                
                # Add score overlay
                h, w = combined_frame.shape[:2]
                overlay = combined_frame.copy()
                
                # Add text background
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                combined_frame = cv2.addWeighted(combined_frame, 0.7, overlay, 0.3, 0)
                
                # Add score text
                score_text = f"Overall Score: {overall_score:.3f} ({results['quality_level'].upper()})"
                cv2.putText(combined_frame, score_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Add individual scores
                score_details = " | ".join([f"{k[:3]}: {v:.2f}" for k, v in results["scores"].items()])
                cv2.putText(combined_frame, score_details, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Save image
                output_path = self.output_dir / f"{overall_score:.3f}_{traj_name}_quality.jpg"
                success = cv2.imwrite(str(output_path), cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
                if success:
                    results["frames_saved"].append(str(output_path))
                    print(f"  Saved visualization to: {output_path}")
                    
                    # Save VLM outputs to JSON
                    json_path = self.output_dir / f"{overall_score:.3f}_{traj_name}_vlm_outputs.json"
                    vlm_data = {
                        "trajectory": traj_name,
                        "overall_score": overall_score,
                        "scores": results["scores"],
                        "explanations": results["explanations"],
                        "full_vlm_responses": vlm_outputs
                    }
                    with open(json_path, 'w') as f:
                        json.dump(vlm_data, f, indent=2)
                    print(f"  Saved VLM outputs to: {json_path}")
                else:
                    print(f"  Failed to save image to: {output_path}")
            except Exception as e:
                print(f"  Error saving visualization: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def run_benchmark(self, max_trajectories: Optional[int] = None) -> Dict[str, Any]:
        """Run the quality scoring benchmark."""
        print("\n" + "=" * 60)
        print("ROBOT DEMONSTRATION QUALITY SCORING")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset(max_trajectories)
        
        # Process trajectories
        process_fn = partial(self.process_single_trajectory)
        results_dataset = dataset.map(process_fn).materialize()
        all_results = list(results_dataset.iter_rows())
        
        # Sort by overall score
        all_results.sort(key=lambda x: x.get("overall_score", 0.0), reverse=True)
        
        # Aggregate statistics
        quality_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        score_statistics = {scorer.get_name(): [] for scorer in self.scorers}
        overall_scores = []
        
        for result in all_results:
            if result.get("quality_level"):
                quality_distribution[result["quality_level"]] += 1
            
            overall_scores.append(result.get("overall_score", 0.0))
            
            for scorer_name, score in result.get("scores", {}).items():
                score_statistics[scorer_name].append(score)
        
        # Print summary
        print("\n" + "=" * 60)
        print("QUALITY SCORING SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal trajectories processed: {len(all_results)}")
        print(f"Average overall score: {np.mean(overall_scores):.3f}")
        print(f"Score range: {np.min(overall_scores):.3f} - {np.max(overall_scores):.3f}")
        
        print("\nQuality Distribution:")
        for level, count in quality_distribution.items():
            percentage = (count / len(all_results)) * 100 if all_results else 0
            print(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("\nComponent Score Statistics:")
        for scorer_name, scores in score_statistics.items():
            if scores:
                print(f"  {scorer_name}:")
                print(f"    Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
                print(f"    Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
        
        # Save detailed results
        results_file = self.output_dir / "quality_scoring_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "config": {
                    "weights": self.scoring_config.weights,
                    "thresholds": self.scoring_config.thresholds,
                    "frames_per_trajectory": self.scoring_config.frames_per_trajectory
                },
                "summary": {
                    "total_trajectories": len(all_results),
                    "average_score": float(np.mean(overall_scores)),
                    "score_range": [float(np.min(overall_scores)), float(np.max(overall_scores))],
                    "quality_distribution": quality_distribution,
                    "component_statistics": {
                        name: {
                            "mean": float(np.mean(scores)),
                            "std": float(np.std(scores)),
                            "min": float(np.min(scores)),
                            "max": float(np.max(scores))
                        } for name, scores in score_statistics.items() if scores
                    }
                },
                "trajectories": all_results
            }, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to {self.output_dir}/")
        print(f"Images saved in order of quality score (highest first)")
        
        return {
            "summary": {
                "total_trajectories": len(all_results),
                "average_score": np.mean(overall_scores),
                "quality_distribution": quality_distribution
            },
            "results": all_results
        }


def main():
    """Main function to run the quality scoring benchmark."""
    parser = argparse.ArgumentParser(description="Score robot demonstration quality using VLM")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./droid_combined_data",
        help="Path to the directory containing VLA trajectory files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quality_scoring_results",
        help="Directory to save scoring results"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=100,
        help="Maximum number of trajectories to process"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to JSON config file for scoring weights and thresholds"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = ScoringConfig()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            if "weights" in config_data:
                config.weights = config_data["weights"]
            if "thresholds" in config_data:
                config.thresholds = config_data["thresholds"]
            if "frames_per_trajectory" in config_data:
                config.frames_per_trajectory = config_data["frames_per_trajectory"]
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create and run benchmark
        benchmark = TrajectoryQualityBenchmark(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            config=config
        )
        
        summary = benchmark.run_benchmark(max_trajectories=args.max_trajectories)
        
        print(f"\nQuality scoring complete!")
        print(f"Average quality score: {summary['summary']['average_score']:.3f}")
        
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()