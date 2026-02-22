#!/usr/bin/env python3
"""
Project 680: Safety-Critical Control Systems for Robotics - Modernized Demo

This is a modernized version of the original safety-critical control system demo,
now using the advanced safety-critical control framework.

DISCLAIMER: This software is for research and educational purposes only.
DO NOT use on real hardware without proper safety review and testing.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits
from safety_critical_control.controllers import (
    ControlBarrierFunctionController,
    ModelPredictiveControlController,
    SafetyControllerManager
)
from safety_critical_control.evaluation import SafetyEvaluator


def run_modernized_demo():
    """Run the modernized safety-critical control demo."""
    print("Safety-Critical Control Systems for Robotics - Modernized Demo")
    print("=" * 60)
    
    # Create safety limits
    safety_limits = SafetyLimits(
        safety_radius=1.0,
        warning_radius=2.0,
        max_velocity=2.0,
        max_acceleration=5.0,
        max_effort=10.0
    )
    
    # Create robot with modernized system
    robot = SafetyCriticalRobot(
        grid_size=(10, 10),
        start_position=(0, 0),
        goal_position=(9, 9),
        safety_limits=safety_limits,
        seed=42
    )
    
    # Create CBF controller
    controller = ControlBarrierFunctionController()
    controller_manager = SafetyControllerManager([controller])
    
    print(f"Robot initialized at position: {robot.position}")
    print(f"Goal position: {robot.goal_position}")
    print(f"Number of obstacles: {len(robot.obstacles)}")
    print(f"Safety radius: {robot.safety_limits.safety_radius}")
    print()
    
    # Run simulation
    step = 0
    max_steps = 1000
    
    print("Starting simulation...")
    print("Step | Position | Safety Status | Distance to Goal")
    print("-" * 50)
    
    while step < max_steps and not robot.is_goal_reached():
        # Get current state
        state = np.concatenate([robot.position, robot.velocity])
        
        # Compute control using CBF controller
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
        
        # Print status every 50 steps
        if step % 50 == 0:
            distance_to_goal = np.linalg.norm(robot.position - robot.goal_position)
            print(f"{step:4d} | ({robot.position[0]:.2f}, {robot.position[1]:.2f}) | {robot.safety_status.value:12s} | {distance_to_goal:.3f}")
        
        # Check for emergency stop
        if not success:
            print(f"\nEmergency stop activated at step {step}!")
            break
        
        step += 1
    
    print("-" * 50)
    
    # Final results
    if robot.is_goal_reached():
        print(f"SUCCESS: Goal reached in {step} steps!")
    else:
        print(f"SIMULATION ENDED: Maximum steps ({max_steps}) reached")
    
    # Get safety metrics
    metrics = robot.get_safety_metrics()
    print(f"\nSafety Metrics:")
    print(f"  Safety violations: {metrics.get('safety_violations', 0)}")
    print(f"  Total distance traveled: {metrics.get('total_distance_traveled', 0):.2f}m")
    print(f"  Path efficiency: {metrics.get('path_efficiency', 0):.3f}")
    print(f"  Final safety status: {robot.safety_status.value}")
    
    # Create visualization
    plot_modernized_results(robot)
    
    return robot, metrics


def plot_modernized_results(robot):
    """Create a modernized visualization of the results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory
    if robot.history:
        trajectory = np.array([state.position for state in robot.history])
        
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        
        # Plot obstacles
        for obstacle in robot.obstacles:
            circle = plt.Circle(obstacle, robot.safety_limits.safety_radius, 
                              color='red', alpha=0.3, label='Safety Zone')
            ax1.add_patch(circle)
            ax1.plot(obstacle[0], obstacle[1], 'kx', markersize=8, label='Obstacle')
        
        ax1.plot(robot.goal_position[0], robot.goal_position[1], 'r*', 
                markersize=15, label='Goal')
        
        ax1.set_xlim(0, robot.grid_size[0])
        ax1.set_ylim(0, robot.grid_size[1])
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Safety-Critical Robot Trajectory')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')
    
    # Plot 2: Safety status over time
    if robot.history:
        timestamps = [state.timestamp for state in robot.history]
        safety_statuses = [state.safety_status.value for state in robot.history]
        
        # Convert safety status to numeric values for plotting
        safety_values = {'safe': 0, 'warning': 1, 'critical': 2, 'emergency_stop': 3}
        numeric_safety = [safety_values.get(status, 0) for status in safety_statuses]
        
        ax2.plot(timestamps, numeric_safety, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Safety Level')
        ax2.set_title('Safety Status Over Time')
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['Safe', 'Warning', 'Critical', 'Emergency'])
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_controllers():
    """Compare different safety controllers."""
    print("\n" + "=" * 60)
    print("CONTROLLER COMPARISON")
    print("=" * 60)
    
    # Create scenarios
    scenarios = [
        {"name": "Easy", "start": (0, 0), "goal": (9, 9), "seed": 42},
        {"name": "Medium", "start": (1, 1), "goal": (8, 8), "seed": 123},
        {"name": "Hard", "start": (2, 2), "goal": (7, 7), "seed": 456}
    ]
    
    controllers = ["CBF", "MPC"]
    evaluator = SafetyEvaluator()
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)
        
        for controller_name in controllers:
            # Create robot
            robot = SafetyCriticalRobot(
                grid_size=(10, 10),
                start_position=scenario["start"],
                goal_position=scenario["goal"],
                safety_limits=SafetyLimits(),
                seed=scenario["seed"]
            )
            
            # Create controller
            if controller_name == "CBF":
                controller = ControlBarrierFunctionController()
            elif controller_name == "MPC":
                controller = ModelPredictiveControlController()
            
            controller_manager = SafetyControllerManager([controller])
            
            # Run simulation
            trajectory = []
            control_history = []
            safety_history = []
            timestamps = []
            
            step = 0
            max_steps = 500
            
            while step < max_steps and not robot.is_goal_reached():
                state = np.concatenate([robot.position, robot.velocity])
                
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
                
                success = robot.move(control)
                
                trajectory.append(robot.position.copy())
                control_history.append(control.copy())
                safety_history.append(robot.safety_status.value)
                timestamps.append(robot.timestamp)
                
                if not success:
                    break
                
                step += 1
            
            # Evaluate results
            metrics = evaluator.evaluate_trajectory(
                trajectory=trajectory,
                obstacles=robot.obstacles,
                goal=robot.goal_position,
                control_history=control_history,
                safety_history=safety_history,
                timestamps=timestamps,
                controller_name=controller_name,
                config={"scenario": scenario["name"]}
            )
            
            print(f"{controller_name:3s}: Success={metrics.success_rate:.3f}, "
                  f"Violations={metrics.safety_violations}, "
                  f"Efficiency={metrics.path_efficiency:.3f}, "
                  f"Time={metrics.completion_time:.2f}s")


if __name__ == "__main__":
    # Run the modernized demo
    robot, metrics = run_modernized_demo()
    
    # Compare controllers
    compare_controllers()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("This modernized version includes:")
    print("- Advanced safety controllers (CBF, MPC)")
    print("- Comprehensive safety metrics")
    print("- Deterministic and reproducible results")
    print("- Type hints and proper documentation")
    print("- Modern Python practices")
    print("\nFor interactive demos, run: streamlit run demo/app.py")
    print("For command-line experiments, run: python scripts/run_simulation.py")