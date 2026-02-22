"""
Unit tests for SafetyCriticalRobot class.
"""

import pytest
import numpy as np
from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits, SafetyStatus


class TestSafetyCriticalRobot:
    """Test cases for SafetyCriticalRobot class."""
    
    def test_robot_initialization(self):
        """Test robot initialization with default parameters."""
        robot = SafetyCriticalRobot()
        
        assert robot.grid_size[0] == 10
        assert robot.grid_size[1] == 10
        assert np.array_equal(robot.position, np.array([0.0, 0.0]))
        assert np.array_equal(robot.goal_position, np.array([9.0, 9.0]))
        assert robot.safety_status == SafetyStatus.SAFE
        assert not robot.emergency_stop_active
    
    def test_robot_initialization_custom(self):
        """Test robot initialization with custom parameters."""
        safety_limits = SafetyLimits(safety_radius=2.0, max_velocity=3.0)
        robot = SafetyCriticalRobot(
            grid_size=(15, 15),
            start_position=(1, 1),
            goal_position=(14, 14),
            safety_limits=safety_limits,
            seed=123
        )
        
        assert robot.grid_size[0] == 15
        assert robot.grid_size[1] == 15
        assert np.array_equal(robot.position, np.array([1.0, 1.0]))
        assert np.array_equal(robot.goal_position, np.array([14.0, 14.0]))
        assert robot.safety_limits.safety_radius == 2.0
        assert robot.safety_limits.max_velocity == 3.0
    
    def test_obstacle_generation(self):
        """Test deterministic obstacle generation."""
        robot1 = SafetyCriticalRobot(seed=42)
        robot2 = SafetyCriticalRobot(seed=42)
        
        # Obstacles should be identical with same seed
        assert len(robot1.obstacles) == len(robot2.obstacles)
        for obs1, obs2 in zip(robot1.obstacles, robot2.obstacles):
            assert np.array_equal(obs1, obs2)
    
    def test_safety_constraint_checking(self):
        """Test safety constraint checking."""
        robot = SafetyCriticalRobot()
        
        # Test safe position
        robot.position = np.array([5.0, 5.0])
        status = robot.check_safety_constraints()
        assert status == SafetyStatus.SAFE
        
        # Test boundary violation
        robot.position = np.array([-1.0, 5.0])
        status = robot.check_safety_constraints()
        assert status == SafetyStatus.EMERGENCY_STOP
        
        # Test obstacle proximity
        robot.position = robot.obstacles[0] + np.array([0.5, 0.5])  # Close to obstacle
        status = robot.check_safety_constraints()
        assert status in [SafetyStatus.WARNING, SafetyStatus.CRITICAL, SafetyStatus.EMERGENCY_STOP]
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        robot = SafetyCriticalRobot()
        
        # Set some velocity
        robot.velocity = np.array([1.0, 1.0])
        
        # Apply emergency stop
        robot.apply_emergency_stop()
        
        assert np.array_equal(robot.velocity, np.array([0.0, 0.0]))
        assert np.array_equal(robot.acceleration, np.array([0.0, 0.0]))
        assert np.array_equal(robot.effort, np.array([0.0, 0.0]))
        assert robot.emergency_stop_active
    
    def test_safe_control_computation(self):
        """Test safe control computation."""
        robot = SafetyCriticalRobot()
        
        # Test with safe desired velocity
        desired_velocity = np.array([0.5, 0.5])
        safe_velocity = robot.compute_safe_control(desired_velocity)
        
        assert np.array_equal(safe_velocity, desired_velocity)
        
        # Test with velocity limit violation
        desired_velocity = np.array([5.0, 5.0])  # Exceeds max_velocity
        safe_velocity = robot.compute_safe_control(desired_velocity)
        
        velocity_magnitude = np.linalg.norm(safe_velocity)
        assert velocity_magnitude <= robot.safety_limits.max_velocity
    
    def test_robot_movement(self):
        """Test robot movement with safety checks."""
        robot = SafetyCriticalRobot()
        initial_position = robot.position.copy()
        
        # Test safe movement
        control_input = np.array([0.1, 0.1])
        success = robot.move(control_input)
        
        assert success
        assert not np.array_equal(robot.position, initial_position)
        assert len(robot.history) > 0
    
    def test_goal_reaching(self):
        """Test goal reaching detection."""
        robot = SafetyCriticalRobot()
        
        # Initially not at goal
        assert not robot.is_goal_reached()
        
        # Move close to goal
        robot.position = robot.goal_position + np.array([0.05, 0.05])
        assert robot.is_goal_reached(tolerance=0.1)
        
        # Move away from goal
        robot.position = robot.goal_position + np.array([0.2, 0.2])
        assert not robot.is_goal_reached(tolerance=0.1)
    
    def test_robot_reset(self):
        """Test robot reset functionality."""
        robot = SafetyCriticalRobot()
        
        # Move robot and modify state
        robot.move(np.array([0.1, 0.1]))
        robot.emergency_stop_active = True
        robot.safety_violations = 5
        
        # Reset robot
        robot.reset()
        
        assert np.array_equal(robot.position, np.array([0.0, 0.0]))
        assert np.array_equal(robot.velocity, np.array([0.0, 0.0]))
        assert robot.safety_status == SafetyStatus.SAFE
        assert not robot.emergency_stop_active
        assert robot.safety_violations == 0
        assert len(robot.history) == 0
    
    def test_safety_metrics(self):
        """Test safety metrics computation."""
        robot = SafetyCriticalRobot()
        
        # Run a short simulation
        for _ in range(10):
            robot.move(np.array([0.1, 0.1]))
        
        metrics = robot.get_safety_metrics()
        
        assert isinstance(metrics, dict)
        assert 'safety_violations' in metrics
        assert 'total_distance_traveled' in metrics
        assert 'path_efficiency' in metrics
        assert metrics['total_distance_traveled'] > 0
    
    def test_cbf_constraints(self):
        """Test Control Barrier Function constraints."""
        robot = SafetyCriticalRobot()
        
        # Position robot close to an obstacle
        robot.position = robot.obstacles[0] + np.array([1.5, 1.5])  # Within warning radius
        
        # Test CBF constraint application
        desired_velocity = np.array([1.0, 1.0])
        safe_velocity = robot._apply_cbf_constraints(desired_velocity)
        
        # Should modify velocity to maintain safety
        assert not np.array_equal(safe_velocity, desired_velocity)
        
        # Check that safety is maintained
        robot.velocity = safe_velocity
        status = robot.check_safety_constraints()
        assert status in [SafetyStatus.SAFE, SafetyStatus.WARNING]
