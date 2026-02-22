"""
Interactive Safety-Critical Control Demo

This module provides an interactive Streamlit-based demo for safety-critical control systems,
allowing users to experiment with different controllers, environments, and safety parameters.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Tuple
import time
import json
from pathlib import Path

# Import our safety-critical control modules
import sys
sys.path.append('src')

from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits, SafetyStatus
from safety_critical_control.controllers import (
    ControlBarrierFunctionController,
    ModelPredictiveControlController,
    EmergencyStopController,
    SafetyControllerManager,
    ControllerConfig
)
from safety_critical_control.evaluation import SafetyEvaluator, ExperimentResult


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'robot' not in st.session_state:
        st.session_state.robot = None
    if 'controller_manager' not in st.session_state:
        st.session_state.controller_manager = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = SafetyEvaluator()
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = []
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False


def create_robot_environment(
    grid_size: Tuple[int, int],
    start_position: Tuple[float, float],
    goal_position: Tuple[float, float],
    safety_limits: SafetyLimits,
    seed: int
) -> SafetyCriticalRobot:
    """Create a new robot environment."""
    return SafetyCriticalRobot(
        grid_size=grid_size,
        start_position=start_position,
        goal_position=goal_position,
        safety_limits=safety_limits,
        seed=seed
    )


def create_controller_manager(controller_types: List[str]) -> SafetyControllerManager:
    """Create controller manager with selected controllers."""
    controllers = []
    
    for controller_type in controller_types:
        config = ControllerConfig()
        
        if controller_type == "CBF":
            controllers.append(ControlBarrierFunctionController(config))
        elif controller_type == "MPC":
            controllers.append(ModelPredictiveControlController(config))
        elif controller_type == "Emergency Stop":
            controllers.append(EmergencyStopController(config))
    
    return SafetyControllerManager(controllers)


def run_simulation(
    robot: SafetyCriticalRobot,
    controller_manager: SafetyControllerManager,
    max_steps: int = 1000,
    dt: float = 0.1
) -> Dict[str, Any]:
    """Run a complete simulation."""
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
    
    return {
        "trajectory": trajectory,
        "control_history": control_history,
        "safety_history": safety_history,
        "timestamps": timestamps,
        "steps": step,
        "simulation_time": simulation_time,
        "goal_reached": robot.is_goal_reached(),
        "final_position": robot.position.copy(),
        "safety_metrics": robot.get_safety_metrics()
    }


def plot_trajectory_2d(
    trajectory: List[np.ndarray],
    obstacles: List[np.ndarray],
    goal: np.ndarray,
    safety_history: List[str],
    title: str = "Robot Trajectory"
) -> go.Figure:
    """Create 2D trajectory plot with safety information."""
    if not trajectory:
        return go.Figure()
    
    # Convert trajectory to arrays
    positions = np.array(trajectory)
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    # Create color map for safety status
    safety_colors = {
        'safe': 'green',
        'warning': 'orange',
        'critical': 'red',
        'emergency_stop': 'darkred'
    }
    
    colors = [safety_colors.get(status, 'blue') for status in safety_history]
    
    # Create the plot
    fig = go.Figure()
    
    # Add trajectory
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color=colors),
        hovertemplate='<b>Position:</b> (%{x:.2f}, %{y:.2f})<br><b>Safety:</b> %{customdata}<extra></extra>',
        customdata=safety_history
    ))
    
    # Add obstacles
    if obstacles:
        obs_positions = np.array(obstacles)
        fig.add_trace(go.Scatter(
            x=obs_positions[:, 0],
            y=obs_positions[:, 1],
            mode='markers',
            name='Obstacles',
            marker=dict(size=15, color='black', symbol='x'),
            hovertemplate='<b>Obstacle</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
    
    # Add goal
    fig.add_trace(go.Scatter(
        x=[goal[0]],
        y=[goal[1]],
        mode='markers',
        name='Goal',
        marker=dict(size=20, color='red', symbol='star'),
        hovertemplate='<b>Goal</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    # Add start position
    fig.add_trace(go.Scatter(
        x=[trajectory[0][0]],
        y=[trajectory[0][1]],
        mode='markers',
        name='Start',
        marker=dict(size=15, color='green', symbol='circle'),
        hovertemplate='<b>Start</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='X Position',
        yaxis_title='Y Position',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def plot_control_signals(
    control_history: List[np.ndarray],
    timestamps: List[float],
    title: str = "Control Signals"
) -> go.Figure:
    """Plot control signals over time."""
    if not control_history or not timestamps:
        return go.Figure()
    
    controls = np.array(control_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Control Input X', 'Control Input Y'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=controls[:, 0], name='Control X', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=controls[:, 1], name='Control Y', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Control Input", row=1, col=1)
    fig.update_yaxes(title_text="Control Input", row=2, col=1)
    
    return fig


def plot_safety_metrics(safety_history: List[str], timestamps: List[float]) -> go.Figure:
    """Plot safety status over time."""
    if not safety_history or not timestamps:
        return go.Figure()
    
    # Convert safety status to numeric values
    safety_values = {
        'safe': 0,
        'warning': 1,
        'critical': 2,
        'emergency_stop': 3
    }
    
    numeric_safety = [safety_values.get(status, 0) for status in safety_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=numeric_safety,
        mode='lines+markers',
        name='Safety Status',
        line=dict(color='red', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Safety:</b> %{customdata}<extra></extra>',
        customdata=safety_history
    ))
    
    fig.update_layout(
        title="Safety Status Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Safety Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Safe', 'Warning', 'Critical', 'Emergency']
        ),
        height=300
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Safety-Critical Control Systems",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Safety-Critical Control Systems for Robotics")
    st.markdown("""
    This interactive demo showcases advanced safety-critical control systems for robotics,
    including Control Barrier Functions (CBF), Model Predictive Control (MPC), and emergency stop systems.
    
    **DISCLAIMER**: This software is for research and educational purposes only. 
    DO NOT use on real hardware without proper safety review and testing.
    """)
    
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment parameters
    st.sidebar.subheader("Environment")
    grid_size_x = st.sidebar.slider("Grid Size X", 5, 20, 10)
    grid_size_y = st.sidebar.slider("Grid Size Y", 5, 20, 10)
    
    start_x = st.sidebar.slider("Start X", 0.0, float(grid_size_x-1), 0.0)
    start_y = st.sidebar.slider("Start Y", 0.0, float(grid_size_y-1), 0.0)
    
    goal_x = st.sidebar.slider("Goal X", 0.0, float(grid_size_x-1), float(grid_size_x-1))
    goal_y = st.sidebar.slider("Goal Y", 0.0, float(grid_size_y-1), float(grid_size_y-1))
    
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Safety parameters
    st.sidebar.subheader("Safety Parameters")
    safety_radius = st.sidebar.slider("Safety Radius", 0.1, 3.0, 1.0)
    warning_radius = st.sidebar.slider("Warning Radius", 0.5, 5.0, 2.0)
    max_velocity = st.sidebar.slider("Max Velocity", 0.1, 5.0, 2.0)
    max_acceleration = st.sidebar.slider("Max Acceleration", 0.1, 10.0, 5.0)
    max_effort = st.sidebar.slider("Max Effort", 0.1, 20.0, 10.0)
    
    # Controller selection
    st.sidebar.subheader("Controllers")
    controller_options = ["CBF", "MPC", "Emergency Stop"]
    selected_controllers = st.sidebar.multiselect(
        "Select Controllers",
        controller_options,
        default=["CBF"]
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation")
    max_steps = st.sidebar.slider("Max Steps", 100, 2000, 1000)
    
    # Create safety limits
    safety_limits = SafetyLimits(
        safety_radius=safety_radius,
        warning_radius=warning_radius,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_effort=max_effort
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Simulation Control")
        
        # Create robot and controller manager
        if st.button("Initialize Robot", key="init_robot"):
            st.session_state.robot = create_robot_environment(
                grid_size=(grid_size_x, grid_size_y),
                start_position=(start_x, start_y),
                goal_position=(goal_x, goal_y),
                safety_limits=safety_limits,
                seed=seed
            )
            
            st.session_state.controller_manager = create_controller_manager(selected_controllers)
            st.success("Robot and controllers initialized!")
        
        # Run simulation
        if st.button("Run Simulation", key="run_sim") and st.session_state.robot is not None:
            with st.spinner("Running simulation..."):
                # Reset robot
                st.session_state.robot.reset((start_x, start_y))
                
                # Run simulation
                simulation_result = run_simulation(
                    st.session_state.robot,
                    st.session_state.controller_manager,
                    max_steps=max_steps
                )
                
                # Store result
                st.session_state.simulation_result = simulation_result
                st.session_state.simulation_running = True
                
                st.success(f"Simulation completed! Steps: {simulation_result['steps']}")
        
        # Display results
        if st.session_state.simulation_running and 'simulation_result' in st.session_state:
            result = st.session_state.simulation_result
            
            # Trajectory plot
            fig_traj = plot_trajectory_2d(
                result['trajectory'],
                st.session_state.robot.obstacles,
                st.session_state.robot.goal_position,
                result['safety_history'],
                "Robot Trajectory with Safety Status"
            )
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Control signals
            fig_control = plot_control_signals(
                result['control_history'],
                result['timestamps'],
                "Control Signals Over Time"
            )
            st.plotly_chart(fig_control, use_container_width=True)
            
            # Safety status
            fig_safety = plot_safety_metrics(
                result['safety_history'],
                result['timestamps']
            )
            st.plotly_chart(fig_safety, use_container_width=True)
    
    with col2:
        st.subheader("Results Summary")
        
        if st.session_state.simulation_running and 'simulation_result' in st.session_state:
            result = st.session_state.simulation_result
            metrics = result['safety_metrics']
            
            # Display key metrics
            st.metric("Goal Reached", "Yes" if result['goal_reached'] else "No")
            st.metric("Steps Taken", result['steps'])
            st.metric("Simulation Time", f"{result['simulation_time']:.3f}s")
            st.metric("Safety Violations", metrics.get('safety_violations', 0))
            st.metric("Path Efficiency", f"{metrics.get('path_efficiency', 0):.3f}")
            st.metric("Total Distance", f"{metrics.get('total_distance_traveled', 0):.2f}m")
            
            # Safety status distribution
            safety_counts = {}
            for status in result['safety_history']:
                safety_counts[status] = safety_counts.get(status, 0) + 1
            
            st.subheader("Safety Status Distribution")
            for status, count in safety_counts.items():
                st.write(f"{status.title()}: {count}")
        
        # Save results
        if st.button("Save Results", key="save_results") and st.session_state.simulation_running:
            # Create experiment result
            if st.session_state.robot is not None and 'simulation_result' in st.session_state:
                result = st.session_state.simulation_result
                
                # Evaluate trajectory
                metrics = st.session_state.evaluator.evaluate_trajectory(
                    trajectory=result['trajectory'],
                    obstacles=st.session_state.robot.obstacles,
                    goal=st.session_state.robot.goal_position,
                    control_history=result['control_history'],
                    safety_history=result['safety_history'],
                    timestamps=result['timestamps'],
                    controller_name="+".join(selected_controllers),
                    config={
                        "safety_radius": safety_radius,
                        "warning_radius": warning_radius,
                        "max_velocity": max_velocity,
                        "max_acceleration": max_acceleration,
                        "max_effort": max_effort,
                        "grid_size": (grid_size_x, grid_size_y),
                        "seed": seed
                    }
                )
                
                # Create experiment result
                exp_result = ExperimentResult(
                    controller_name="+".join(selected_controllers),
                    metrics=metrics,
                    trajectory=result['trajectory'],
                    control_history=result['control_history'],
                    safety_history=result['safety_history'],
                    timestamps=result['timestamps'],
                    config={
                        "safety_radius": safety_radius,
                        "warning_radius": warning_radius,
                        "max_velocity": max_velocity,
                        "max_acceleration": max_acceleration,
                        "max_effort": max_effort,
                        "grid_size": (grid_size_x, grid_size_y),
                        "seed": seed
                    }
                )
                
                # Add to evaluator
                st.session_state.evaluator.add_experiment_result(exp_result)
                st.session_state.experiment_results.append(exp_result)
                
                st.success("Results saved!")
    
    # Evaluation section
    st.subheader("Comparative Evaluation")
    
    if st.session_state.experiment_results:
        # Generate leaderboard
        leaderboard = st.session_state.evaluator.generate_leaderboard()
        
        # Display rankings
        st.subheader("Controller Rankings")
        rankings_df = pd.DataFrame(leaderboard['rankings'])
        st.dataframe(rankings_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        summary_df = pd.DataFrame([leaderboard['summary_stats']])
        st.dataframe(summary_df, use_container_width=True)
        
        # Generate report
        if st.button("Generate Report", key="generate_report"):
            report = st.session_state.evaluator.generate_report()
            st.text_area("Evaluation Report", report, height=400)
        
        # Save all results
        if st.button("Save All Results", key="save_all"):
            results_path = "assets/evaluation_results.json"
            Path("assets").mkdir(exist_ok=True)
            st.session_state.evaluator.save_results(results_path)
            st.success(f"All results saved to {results_path}")
    
    else:
        st.info("Run simulations and save results to see comparative evaluation.")


if __name__ == "__main__":
    main()
