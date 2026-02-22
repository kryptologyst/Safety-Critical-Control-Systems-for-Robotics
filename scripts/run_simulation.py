#!/usr/bin/env python3
"""
Safety-Critical Control Systems - Main Simulation Script

This script demonstrates the safety-critical control systems with different controllers
and generates comprehensive evaluation results.

Usage:
    python scripts/run_simulation.py --controller CBF --scenarios 5
    python scripts/run_simulation.py --controller MPC --scenarios 10 --save-results
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any

# Add src to path
import sys
sys.path.append('src')

from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits
from safety_critical_control.controllers import (
    ControlBarrierFunctionController,
    ModelPredictiveControlController,
    EmergencyStopController,
    SafetyControllerManager,
    ControllerConfig
)
from safety_critical_control.evaluation import SafetyEvaluator, ExperimentResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_scenario(
    scenario_id: int,
    grid_size: tuple = (10, 10),
    seed: int = 42
) -> Dict[str, Any]:
    """Create a test scenario."""
    np.random.seed(seed + scenario_id)
    
    # Generate start and goal positions
    start_pos = (0.0, 0.0)
    goal_pos = (float(grid_size[0] - 1), float(grid_size[1] - 1))
    
    # Add some variation
    if scenario_id > 0:
        start_pos = (
            np.random.uniform(0, grid_size[0] * 0.3),
            np.random.uniform(0, grid_size[1] * 0.3)
        )
        goal_pos = (
            np.random.uniform(grid_size[0] * 0.7, grid_size[0]),
            np.random.uniform(grid_size[1] * 0.7, grid_size[1])
        )
    
    return {
        "scenario_id": scenario_id,
        "grid_size": grid_size,
        "start_position": start_pos,
        "goal_position": goal_pos,
        "seed": seed + scenario_id
    }


def run_controller_experiment(
    controller_name: str,
    scenario: Dict[str, Any],
    safety_limits: SafetyLimits,
    max_steps: int = 1000
) -> ExperimentResult:
    """Run a single experiment with a specific controller."""
    logger.info(f"Running experiment: {controller_name} on scenario {scenario['scenario_id']}")
    
    # Create robot
    robot = SafetyCriticalRobot(
        grid_size=scenario["grid_size"],
        start_position=scenario["start_position"],
        goal_position=scenario["goal_position"],
        safety_limits=safety_limits,
        seed=scenario["seed"]
    )
    
    # Create controller
    config = ControllerConfig()
    if controller_name == "CBF":
        controller = ControlBarrierFunctionController(config)
    elif controller_name == "MPC":
        controller = ModelPredictiveControlController(config)
    elif controller_name == "Emergency Stop":
        controller = EmergencyStopController(config)
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    
    controller_manager = SafetyControllerManager([controller])
    
    # Run simulation
    trajectory = []
    control_history = []
    safety_history = []
    timestamps = []
    
    step = 0
    start_time = time.time()
    
    while step < max_steps and not robot.is_goal_reached():
        # Get current state
        state = np.concatenate([robot.position, robot.velocity])
        
        # Compute control
        control, safety_info = controller_manager.compute_control(
            state=state,
            reference=robot.goal_position,
            obstacles=robot.obstacles,
            safety_limits={
                "safety_radius": robot.safety_limits.safety_radius,
                "warning_radius": robot.safety_limits.warning_radius,
                "max_velocity": robot.safety_limits.max_velocity,
                "max_acceleration": robot.safety_limits.max_acceleration,
                "max_effort": robot.safety_limits.max_effort
            }
        )
        
        # Move robot
        success = robot.move(control)
        
        # Record data
        trajectory.append(robot.position.copy())
        control_history.append(control.copy())
        safety_history.append(robot.safety_status.value)
        timestamps.append(robot.timestamp)
        
        # Check for emergency stop
        if not success:
            break
        
        step += 1
    
    simulation_time = time.time() - start_time
    
    # Evaluate trajectory
    evaluator = SafetyEvaluator()
    metrics = evaluator.evaluate_trajectory(
        trajectory=trajectory,
        obstacles=robot.obstacles,
        goal=robot.goal_position,
        control_history=control_history,
        safety_history=safety_history,
        timestamps=timestamps,
        controller_name=controller_name,
        config={
            "scenario_id": scenario["scenario_id"],
            "grid_size": scenario["grid_size"],
            "seed": scenario["seed"],
            "max_steps": max_steps,
            "simulation_time": simulation_time
        }
    )
    
    # Create experiment result
    result = ExperimentResult(
        controller_name=controller_name,
        metrics=metrics,
        trajectory=trajectory,
        control_history=control_history,
        safety_history=safety_history,
        timestamps=timestamps,
        config={
            "scenario_id": scenario["scenario_id"],
            "grid_size": scenario["grid_size"],
            "seed": scenario["seed"],
            "max_steps": max_steps,
            "simulation_time": simulation_time
        }
    )
    
    logger.info(f"Experiment completed: {controller_name} - Steps: {step}, Goal reached: {robot.is_goal_reached()}")
    
    return result


def plot_comparison(results: List[ExperimentResult], save_path: str = None):
    """Plot comparison of different controllers."""
    if not results:
        logger.warning("No results to plot")
        return
    
    # Group results by controller
    controller_results = {}
    for result in results:
        if result.controller_name not in controller_results:
            controller_results[result.controller_name] = []
        controller_results[result.controller_name].append(result)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Safety-Critical Control Performance Comparison', fontsize=16)
    
    controllers = list(controller_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(controllers)))
    
    # Success rate
    success_rates = [np.mean([r.metrics.success_rate for r in controller_results[c]]) for c in controllers]
    axes[0, 0].bar(controllers, success_rates, color=colors, alpha=0.7)
    axes[0, 0].set_title('Average Success Rate')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Safety violations
    safety_violations = [np.mean([r.metrics.safety_violations for r in controller_results[c]]) for c in controllers]
    axes[0, 1].bar(controllers, safety_violations, color=colors, alpha=0.7)
    axes[0, 1].set_title('Average Safety Violations')
    axes[0, 1].set_ylabel('Number of Violations')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Path efficiency
    path_efficiencies = [np.mean([r.metrics.path_efficiency for r in controller_results[c]]) for c in controllers]
    axes[0, 2].bar(controllers, path_efficiencies, color=colors, alpha=0.7)
    axes[0, 2].set_title('Average Path Efficiency')
    axes[0, 2].set_ylabel('Efficiency')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Completion time
    completion_times = [np.mean([r.metrics.completion_time for r in controller_results[c]]) for c in controllers]
    axes[1, 0].bar(controllers, completion_times, color=colors, alpha=0.7)
    axes[1, 0].set_title('Average Completion Time')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Safety margin
    safety_margins = [np.mean([r.metrics.safety_margin_min for r in controller_results[c]]) for c in controllers]
    axes[1, 1].bar(controllers, safety_margins, color=colors, alpha=0.7)
    axes[1, 1].set_title('Average Minimum Safety Margin')
    axes[1, 1].set_ylabel('Distance (m)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Control effort
    control_efforts = [np.mean([r.metrics.control_effort for r in controller_results[c]]) for c in controllers]
    axes[1, 2].bar(controllers, control_efforts, color=colors, alpha=0.7)
    axes[1, 2].set_title('Average Control Effort')
    axes[1, 2].set_ylabel('Effort')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run safety-critical control experiments')
    parser.add_argument('--controller', type=str, default='CBF',
                       choices=['CBF', 'MPC', 'Emergency Stop'],
                       help='Controller to test')
    parser.add_argument('--scenarios', type=int, default=5,
                       help='Number of scenarios to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum simulation steps')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output-dir', type=str, default='assets',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Safety limits
    safety_limits = SafetyLimits(
        safety_radius=1.0,
        warning_radius=2.0,
        max_velocity=2.0,
        max_acceleration=5.0,
        max_effort=10.0
    )
    
    # Create scenarios
    scenarios = [create_scenario(i) for i in range(args.scenarios)]
    
    # Run experiments
    results = []
    evaluator = SafetyEvaluator()
    
    logger.info(f"Running {args.scenarios} experiments with {args.controller} controller")
    
    for scenario in scenarios:
        result = run_controller_experiment(
            controller_name=args.controller,
            scenario=scenario,
            safety_limits=safety_limits,
            max_steps=args.max_steps
        )
        
        results.append(result)
        evaluator.add_experiment_result(result)
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    leaderboard = evaluator.generate_leaderboard()
    
    print("\n" + "="*60)
    print("SAFETY-CRITICAL CONTROL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nController: {args.controller}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Max Steps: {args.max_steps}")
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Average Success Rate: {leaderboard['summary_stats']['avg_success_rate']:.3f}")
    print(f"Average Safety Violations: {leaderboard['summary_stats']['avg_safety_violations']:.1f}")
    print(f"Average Path Efficiency: {leaderboard['summary_stats']['avg_path_efficiency']:.3f}")
    print(f"Average Completion Time: {leaderboard['summary_stats']['avg_completion_time']:.2f}s")
    print(f"Best Safety Margin: {leaderboard['summary_stats']['best_safety_margin']:.3f}m")
    
    print(f"\nDETAILED RESULTS:")
    for i, result in enumerate(results):
        print(f"\nScenario {i+1}:")
        print(f"  Success Rate: {result.metrics.success_rate:.3f}")
        print(f"  Safety Violations: {result.metrics.safety_violations}")
        print(f"  Path Efficiency: {result.metrics.path_efficiency:.3f}")
        print(f"  Completion Time: {result.metrics.completion_time:.2f}s")
        print(f"  Min Safety Margin: {result.metrics.safety_margin_min:.3f}m")
        print(f"  Control Effort: {result.metrics.control_effort:.3f}")
    
    # Save results
    if args.save_results:
        results_file = output_dir / f"{args.controller.lower().replace(' ', '_')}_results.json"
        evaluator.save_results(str(results_file))
        logger.info(f"Results saved to {results_file}")
        
        # Save detailed report
        report_file = output_dir / f"{args.controller.lower().replace(' ', '_')}_report.txt"
        report = evaluator.generate_report()
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")
    
    # Plot comparison (if multiple controllers)
    if len(set(r.controller_name for r in results)) > 1:
        plot_file = output_dir / f"{args.controller.lower().replace(' ', '_')}_comparison.png"
        plot_comparison(results, str(plot_file))
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
