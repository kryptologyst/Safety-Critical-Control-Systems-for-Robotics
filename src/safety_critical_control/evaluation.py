"""
Safety-Critical Control Evaluation Metrics

This module implements comprehensive evaluation metrics for safety-critical control systems,
including safety performance benchmarks, trajectory analysis, and comparative evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety level classification."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics."""
    # Safety violations
    safety_violations: int = 0
    critical_violations: int = 0
    emergency_stops: int = 0
    
    # Distance metrics
    min_obstacle_distance: float = float('inf')
    avg_obstacle_distance: float = 0.0
    safety_margin_violations: int = 0
    
    # Trajectory metrics
    path_length: float = 0.0
    path_efficiency: float = 0.0
    trajectory_smoothness: float = 0.0
    
    # Control metrics
    control_effort: float = 0.0
    control_smoothness: float = 0.0
    max_velocity: float = 0.0
    max_acceleration: float = 0.0
    
    # Performance metrics
    goal_reached: bool = False
    completion_time: float = 0.0
    success_rate: float = 0.0
    
    # Safety margins
    safety_margin_mean: float = 0.0
    safety_margin_std: float = 0.0
    safety_margin_min: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "safety_violations": self.safety_violations,
            "critical_violations": self.critical_violations,
            "emergency_stops": self.emergency_stops,
            "min_obstacle_distance": self.min_obstacle_distance,
            "avg_obstacle_distance": self.avg_obstacle_distance,
            "safety_margin_violations": self.safety_margin_violations,
            "path_length": self.path_length,
            "path_efficiency": self.path_efficiency,
            "trajectory_smoothness": self.trajectory_smoothness,
            "control_effort": self.control_effort,
            "control_smoothness": self.control_smoothness,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "goal_reached": self.goal_reached,
            "completion_time": self.completion_time,
            "success_rate": self.success_rate,
            "safety_margin_mean": self.safety_margin_mean,
            "safety_margin_std": self.safety_margin_std,
            "safety_margin_min": self.safety_margin_min
        }


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    controller_name: str
    metrics: SafetyMetrics
    trajectory: List[np.ndarray]
    control_history: List[np.ndarray]
    safety_history: List[str]
    timestamps: List[float]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment result to dictionary."""
        return {
            "controller_name": self.controller_name,
            "metrics": self.metrics.to_dict(),
            "trajectory": [pos.tolist() for pos in self.trajectory],
            "control_history": [ctrl.tolist() for ctrl in self.control_history],
            "safety_history": self.safety_history,
            "timestamps": self.timestamps,
            "config": self.config
        }


class SafetyEvaluator:
    """
    Comprehensive safety evaluation system.
    
    Evaluates safety-critical control systems using multiple metrics
    and provides comparative analysis capabilities.
    """
    
    def __init__(self, safety_radius: float = 1.0, warning_radius: float = 2.0):
        """
        Initialize safety evaluator.
        
        Args:
            safety_radius: Minimum safe distance to obstacles
            warning_radius: Warning distance threshold
        """
        self.safety_radius = safety_radius
        self.warning_radius = warning_radius
        self.results: List[ExperimentResult] = []
        logger.info("Safety evaluator initialized")
    
    def evaluate_trajectory(
        self,
        trajectory: List[np.ndarray],
        obstacles: List[np.ndarray],
        goal: np.ndarray,
        control_history: List[np.ndarray],
        safety_history: List[str],
        timestamps: List[float],
        controller_name: str,
        config: Dict[str, Any]
    ) -> SafetyMetrics:
        """
        Evaluate a single trajectory for safety metrics.
        
        Args:
            trajectory: List of robot positions
            obstacles: List of obstacle positions
            goal: Goal position
            control_history: List of control inputs
            safety_history: List of safety status strings
            timestamps: List of timestamps
            controller_name: Name of the controller
            config: Configuration used
            
        Returns:
            Comprehensive safety metrics
        """
        if not trajectory or not timestamps:
            return SafetyMetrics()
        
        metrics = SafetyMetrics()
        
        # Convert safety history to safety levels
        safety_levels = [self._parse_safety_status(status) for status in safety_history]
        
        # Safety violation analysis
        metrics.safety_violations = sum(1 for level in safety_levels 
                                      if level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY])
        metrics.critical_violations = sum(1 for level in safety_levels 
                                        if level == SafetyLevel.CRITICAL)
        metrics.emergency_stops = sum(1 for level in safety_levels 
                                     if level == SafetyLevel.EMERGENCY)
        
        # Distance analysis
        obstacle_distances = []
        safety_margins = []
        
        for pos in trajectory:
            min_dist = min(np.linalg.norm(pos - obs) for obs in obstacles) if obstacles else float('inf')
            obstacle_distances.append(min_dist)
            safety_margins.append(min_dist - self.safety_radius)
        
        metrics.min_obstacle_distance = min(obstacle_distances) if obstacle_distances else float('inf')
        metrics.avg_obstacle_distance = np.mean(obstacle_distances) if obstacle_distances else 0.0
        metrics.safety_margin_violations = sum(1 for margin in safety_margins if margin < 0)
        
        # Safety margin statistics
        if safety_margins:
            metrics.safety_margin_mean = np.mean(safety_margins)
            metrics.safety_margin_std = np.std(safety_margins)
            metrics.safety_margin_min = min(safety_margins)
        
        # Trajectory analysis
        metrics.path_length = self._compute_path_length(trajectory)
        metrics.path_efficiency = self._compute_path_efficiency(trajectory, goal)
        metrics.trajectory_smoothness = self._compute_trajectory_smoothness(trajectory)
        
        # Control analysis
        if control_history:
            metrics.control_effort = self._compute_control_effort(control_history)
            metrics.control_smoothness = self._compute_control_smoothness(control_history)
            metrics.max_velocity = max(np.linalg.norm(ctrl) for ctrl in control_history)
        
        # Performance metrics
        metrics.goal_reached = self._check_goal_reached(trajectory[-1], goal)
        metrics.completion_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Success rate (1.0 if goal reached without critical violations)
        if metrics.goal_reached and metrics.critical_violations == 0:
            metrics.success_rate = 1.0
        elif metrics.goal_reached:
            metrics.success_rate = 0.5  # Partial success
        else:
            metrics.success_rate = 0.0
        
        return metrics
    
    def _parse_safety_status(self, status: str) -> SafetyLevel:
        """Parse safety status string to SafetyLevel enum."""
        status_lower = status.lower()
        if "emergency" in status_lower:
            return SafetyLevel.EMERGENCY
        elif "critical" in status_lower:
            return SafetyLevel.CRITICAL
        elif "warning" in status_lower:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE
    
    def _compute_path_length(self, trajectory: List[np.ndarray]) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            total_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        return total_length
    
    def _compute_path_efficiency(self, trajectory: List[np.ndarray], goal: np.ndarray) -> float:
        """Compute path efficiency (straight-line distance / actual path length)."""
        if not trajectory:
            return 0.0
        
        straight_line_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        path_length = self._compute_path_length(trajectory)
        
        return straight_line_distance / path_length if path_length > 0 else 0.0
    
    def _compute_trajectory_smoothness(self, trajectory: List[np.ndarray]) -> float:
        """Compute trajectory smoothness using curvature."""
        if len(trajectory) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Compute curvature using cross product
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                curvature = abs(np.cross(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))**2
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _compute_control_effort(self, control_history: List[np.ndarray]) -> float:
        """Compute total control effort."""
        return sum(np.linalg.norm(ctrl)**2 for ctrl in control_history)
    
    def _compute_control_smoothness(self, control_history: List[np.ndarray]) -> float:
        """Compute control smoothness using jerk."""
        if len(control_history) < 2:
            return 0.0
        
        jerks = []
        for i in range(1, len(control_history)):
            jerk = np.linalg.norm(control_history[i] - control_history[i-1])
            jerks.append(jerk)
        
        return np.mean(jerks) if jerks else 0.0
    
    def _check_goal_reached(self, final_position: np.ndarray, goal: np.ndarray, tolerance: float = 0.1) -> bool:
        """Check if goal was reached."""
        return np.linalg.norm(final_position - goal) < tolerance
    
    def add_experiment_result(self, result: ExperimentResult) -> None:
        """Add an experiment result to the evaluator."""
        self.results.append(result)
        logger.info(f"Added experiment result for {result.controller_name}")
    
    def generate_leaderboard(self) -> Dict[str, Any]:
        """
        Generate a comprehensive leaderboard of all experiments.
        
        Returns:
            Dictionary containing leaderboard data
        """
        if not self.results:
            return {"error": "No experiment results available"}
        
        # Sort results by success rate, then by safety violations
        sorted_results = sorted(
            self.results,
            key=lambda r: (r.metrics.success_rate, -r.metrics.safety_violations),
            reverse=True
        )
        
        leaderboard = {
            "rankings": [],
            "summary_stats": {},
            "comparative_analysis": {}
        }
        
        # Generate rankings
        for i, result in enumerate(sorted_results):
            ranking = {
                "rank": i + 1,
                "controller": result.controller_name,
                "success_rate": result.metrics.success_rate,
                "safety_violations": result.metrics.safety_violations,
                "path_efficiency": result.metrics.path_efficiency,
                "completion_time": result.metrics.completion_time,
                "safety_margin_min": result.metrics.safety_margin_min,
                "control_effort": result.metrics.control_effort
            }
            leaderboard["rankings"].append(ranking)
        
        # Summary statistics
        all_metrics = [r.metrics for r in self.results]
        leaderboard["summary_stats"] = {
            "total_experiments": len(self.results),
            "avg_success_rate": np.mean([m.success_rate for m in all_metrics]),
            "avg_safety_violations": np.mean([m.safety_violations for m in all_metrics]),
            "avg_path_efficiency": np.mean([m.path_efficiency for m in all_metrics]),
            "avg_completion_time": np.mean([m.completion_time for m in all_metrics]),
            "best_safety_margin": max([m.safety_margin_min for m in all_metrics]),
            "lowest_control_effort": min([m.control_effort for m in all_metrics])
        }
        
        # Comparative analysis
        controller_names = list(set(r.controller_name for r in self.results))
        leaderboard["comparative_analysis"] = {
            "controller_comparison": {}
        }
        
        for controller in controller_names:
            controller_results = [r for r in self.results if r.controller_name == controller]
            controller_metrics = [r.metrics for r in controller_results]
            
            leaderboard["comparative_analysis"]["controller_comparison"][controller] = {
                "experiment_count": len(controller_results),
                "avg_success_rate": np.mean([m.success_rate for m in controller_metrics]),
                "avg_safety_violations": np.mean([m.safety_violations for m in controller_metrics]),
                "avg_path_efficiency": np.mean([m.path_efficiency for m in controller_metrics]),
                "avg_completion_time": np.mean([m.completion_time for m in controller_metrics]),
                "avg_safety_margin_min": np.mean([m.safety_margin_min for m in controller_metrics]),
                "avg_control_effort": np.mean([m.control_effort for m in controller_metrics])
            }
        
        return leaderboard
    
    def save_results(self, filepath: str) -> None:
        """Save all experiment results to a JSON file."""
        results_data = {
            "evaluator_config": {
                "safety_radius": self.safety_radius,
                "warning_radius": self.warning_radius
            },
            "experiments": [result.to_dict() for result in self.results],
            "leaderboard": self.generate_leaderboard()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load experiment results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing results
        self.results = []
        
        # Load experiments
        for exp_data in data.get("experiments", []):
            # Reconstruct trajectory and control history
            trajectory = [np.array(pos) for pos in exp_data["trajectory"]]
            control_history = [np.array(ctrl) for ctrl in exp_data["control_history"]]
            
            # Reconstruct metrics
            metrics = SafetyMetrics(**exp_data["metrics"])
            
            result = ExperimentResult(
                controller_name=exp_data["controller_name"],
                metrics=metrics,
                trajectory=trajectory,
                control_history=control_history,
                safety_history=exp_data["safety_history"],
                timestamps=exp_data["timestamps"],
                config=exp_data["config"]
            )
            
            self.results.append(result)
        
        logger.info(f"Loaded {len(self.results)} experiment results from {filepath}")
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comparative analysis of all experiments."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Safety-Critical Control Performance Comparison', fontsize=16)
        
        # Extract data for plotting
        controllers = [r.controller_name for r in self.results]
        success_rates = [r.metrics.success_rate for r in self.results]
        safety_violations = [r.metrics.safety_violations for r in self.results]
        path_efficiencies = [r.metrics.path_efficiency for r in self.results]
        completion_times = [r.metrics.completion_time for r in self.results]
        safety_margins = [r.metrics.safety_margin_min for r in self.results]
        control_efforts = [r.metrics.control_effort for r in self.results]
        
        # Success Rate
        axes[0, 0].bar(controllers, success_rates, color='green', alpha=0.7)
        axes[0, 0].set_title('Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Safety Violations
        axes[0, 1].bar(controllers, safety_violations, color='red', alpha=0.7)
        axes[0, 1].set_title('Safety Violations')
        axes[0, 1].set_ylabel('Number of Violations')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Path Efficiency
        axes[0, 2].bar(controllers, path_efficiencies, color='blue', alpha=0.7)
        axes[0, 2].set_title('Path Efficiency')
        axes[0, 2].set_ylabel('Efficiency')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Completion Time
        axes[1, 0].bar(controllers, completion_times, color='orange', alpha=0.7)
        axes[1, 0].set_title('Completion Time')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Safety Margin
        axes[1, 1].bar(controllers, safety_margins, color='purple', alpha=0.7)
        axes[1, 1].set_title('Minimum Safety Margin')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Control Effort
        axes[1, 2].bar(controllers, control_efforts, color='brown', alpha=0.7)
        axes[1, 2].set_title('Control Effort')
        axes[1, 2].set_ylabel('Effort')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        if not self.results:
            return "No experiment results available for report generation."
        
        leaderboard = self.generate_leaderboard()
        
        report = f"""
SAFETY-CRITICAL CONTROL EVALUATION REPORT
==========================================

SUMMARY STATISTICS
------------------
Total Experiments: {leaderboard['summary_stats']['total_experiments']}
Average Success Rate: {leaderboard['summary_stats']['avg_success_rate']:.3f}
Average Safety Violations: {leaderboard['summary_stats']['avg_safety_violations']:.1f}
Average Path Efficiency: {leaderboard['summary_stats']['avg_path_efficiency']:.3f}
Average Completion Time: {leaderboard['summary_stats']['avg_completion_time']:.2f}s
Best Safety Margin: {leaderboard['summary_stats']['best_safety_margin']:.3f}m
Lowest Control Effort: {leaderboard['summary_stats']['lowest_control_effort']:.3f}

CONTROLLER RANKINGS
-------------------
"""
        
        for ranking in leaderboard['rankings']:
            report += f"""
Rank {ranking['rank']}: {ranking['controller']}
  Success Rate: {ranking['success_rate']:.3f}
  Safety Violations: {ranking['safety_violations']}
  Path Efficiency: {ranking['path_efficiency']:.3f}
  Completion Time: {ranking['completion_time']:.2f}s
  Min Safety Margin: {ranking['safety_margin_min']:.3f}m
  Control Effort: {ranking['control_effort']:.3f}
"""
        
        report += "\nCONTROLLER COMPARISON\n-------------------\n"
        
        for controller, stats in leaderboard['comparative_analysis']['controller_comparison'].items():
            report += f"""
{controller}:
  Experiments: {stats['experiment_count']}
  Avg Success Rate: {stats['avg_success_rate']:.3f}
  Avg Safety Violations: {stats['avg_safety_violations']:.1f}
  Avg Path Efficiency: {stats['avg_path_efficiency']:.3f}
  Avg Completion Time: {stats['avg_completion_time']:.2f}s
  Avg Min Safety Margin: {stats['avg_safety_margin_min']:.3f}m
  Avg Control Effort: {stats['avg_control_effort']:.3f}
"""
        
        return report
