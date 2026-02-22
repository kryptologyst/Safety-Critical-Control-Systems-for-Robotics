"""
Safety-Critical Robot Environment

This module implements a modernized safety-critical robot environment with:
- Deterministic obstacle generation
- Safety constraint checking
- Emergency stop mechanisms
- Collision avoidance
- Velocity and effort limits
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Safety status enumeration."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    max_velocity: float = 2.0
    max_acceleration: float = 5.0
    safety_radius: float = 1.0
    warning_radius: float = 2.0
    emergency_stop_distance: float = 0.5
    max_effort: float = 10.0


@dataclass
class RobotState:
    """Robot state representation."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    effort: np.ndarray
    safety_status: SafetyStatus
    timestamp: float


class SafetyCriticalRobot:
    """
    Modernized safety-critical robot with advanced safety mechanisms.
    
    This robot implements:
    - Control Barrier Functions (CBF) for safety
    - Emergency stop systems
    - Velocity and effort limits
    - Collision avoidance
    - Safety status monitoring
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        start_position: Tuple[float, float] = (0.0, 0.0),
        goal_position: Tuple[float, float] = (9.0, 9.0),
        safety_limits: Optional[SafetyLimits] = None,
        seed: int = 42
    ) -> None:
        """
        Initialize the safety-critical robot.
        
        Args:
            grid_size: Size of the environment grid
            start_position: Initial robot position
            goal_position: Target goal position
            safety_limits: Safety limits configuration
            seed: Random seed for reproducibility
        """
        self.grid_size = np.array(grid_size, dtype=float)
        self.position = np.array(start_position, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)
        self.safety_limits = safety_limits or SafetyLimits()
        
        # Robot dynamics
        self.velocity = np.zeros(2, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.effort = np.zeros(2, dtype=float)
        
        # Safety state
        self.safety_status = SafetyStatus.SAFE
        self.emergency_stop_active = False
        self.timestamp = 0.0
        
        # Environment
        self.obstacles = self._generate_obstacles(seed)
        self.history: List[RobotState] = []
        
        # Safety monitoring
        self.safety_violations = 0
        self.total_distance_traveled = 0.0
        
        logger.info(f"Safety-critical robot initialized at {self.position}")
        logger.info(f"Goal position: {self.goal_position}")
        logger.info(f"Safety limits: {self.safety_limits}")
    
    def _generate_obstacles(self, seed: int) -> List[np.ndarray]:
        """
        Generate deterministic obstacles for reproducible experiments.
        
        Args:
            seed: Random seed for obstacle generation
            
        Returns:
            List of obstacle positions
        """
        np.random.seed(seed)
        obstacles = []
        
        # Generate obstacles avoiding start and goal positions
        for _ in range(5):
            while True:
                x = np.random.uniform(1, self.grid_size[0] - 1)
                y = np.random.uniform(1, self.grid_size[1] - 1)
                obstacle_pos = np.array([x, y])
                
                # Ensure obstacle is not too close to start or goal
                start_dist = np.linalg.norm(obstacle_pos - self.position)
                goal_dist = np.linalg.norm(obstacle_pos - self.goal_position)
                
                if start_dist > 2.0 and goal_dist > 2.0:
                    obstacles.append(obstacle_pos)
                    break
        
        logger.info(f"Generated {len(obstacles)} obstacles")
        return obstacles
    
    def check_safety_constraints(self) -> SafetyStatus:
        """
        Check all safety constraints and return current safety status.
        
        Returns:
            Current safety status
        """
        # Check velocity limits
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > self.safety_limits.max_velocity:
            logger.warning(f"Velocity limit exceeded: {velocity_magnitude:.2f} > {self.safety_limits.max_velocity}")
            return SafetyStatus.CRITICAL
        
        # Check acceleration limits
        acceleration_magnitude = np.linalg.norm(self.acceleration)
        if acceleration_magnitude > self.safety_limits.max_acceleration:
            logger.warning(f"Acceleration limit exceeded: {acceleration_magnitude:.2f} > {self.safety_limits.max_acceleration}")
            return SafetyStatus.CRITICAL
        
        # Check effort limits
        effort_magnitude = np.linalg.norm(self.effort)
        if effort_magnitude > self.safety_limits.max_effort:
            logger.warning(f"Effort limit exceeded: {effort_magnitude:.2f} > {self.safety_limits.max_effort}")
            return SafetyStatus.CRITICAL
        
        # Check proximity to obstacles
        min_obstacle_distance = float('inf')
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.position - obstacle)
            min_obstacle_distance = min(min_obstacle_distance, distance)
            
            if distance < self.safety_limits.emergency_stop_distance:
                logger.error(f"Emergency stop triggered! Distance to obstacle: {distance:.2f}")
                return SafetyStatus.EMERGENCY_STOP
            elif distance < self.safety_limits.safety_radius:
                logger.warning(f"Safety radius violated! Distance to obstacle: {distance:.2f}")
                return SafetyStatus.CRITICAL
            elif distance < self.safety_limits.warning_radius:
                return SafetyStatus.WARNING
        
        # Check boundary constraints
        if np.any(self.position < 0) or np.any(self.position >= self.grid_size):
            logger.error("Robot out of bounds!")
            return SafetyStatus.EMERGENCY_STOP
        
        return SafetyStatus.SAFE
    
    def apply_emergency_stop(self) -> None:
        """Apply emergency stop by setting all velocities to zero."""
        self.velocity = np.zeros_like(self.velocity)
        self.acceleration = np.zeros_like(self.acceleration)
        self.effort = np.zeros_like(self.effort)
        self.emergency_stop_active = True
        logger.critical("EMERGENCY STOP ACTIVATED!")
    
    def compute_safe_control(self, desired_velocity: np.ndarray) -> np.ndarray:
        """
        Compute safe control input using Control Barrier Functions.
        
        Args:
            desired_velocity: Desired velocity from high-level controller
            
        Returns:
            Safe velocity command
        """
        if self.emergency_stop_active:
            return np.zeros_like(desired_velocity)
        
        # Apply velocity limits
        velocity_magnitude = np.linalg.norm(desired_velocity)
        if velocity_magnitude > self.safety_limits.max_velocity:
            desired_velocity = desired_velocity / velocity_magnitude * self.safety_limits.max_velocity
        
        # Apply CBF-based safety constraints
        safe_velocity = self._apply_cbf_constraints(desired_velocity)
        
        return safe_velocity
    
    def _apply_cbf_constraints(self, desired_velocity: np.ndarray) -> np.ndarray:
        """
        Apply Control Barrier Function constraints for collision avoidance.
        
        Args:
            desired_velocity: Desired velocity
            
        Returns:
            Safe velocity that satisfies CBF constraints
        """
        safe_velocity = desired_velocity.copy()
        
        # CBF constraint for each obstacle
        for obstacle in self.obstacles:
            # Distance to obstacle
            distance = np.linalg.norm(self.position - obstacle)
            
            if distance < self.safety_limits.warning_radius:
                # Compute barrier function gradient
                barrier_gradient = (self.position - obstacle) / distance
                
                # Compute safe velocity that maintains safety
                # h_dot >= -alpha * h, where h = distance - safety_radius
                h = distance - self.safety_limits.safety_radius
                alpha = 1.0  # CBF parameter
                
                # Constraint: barrier_gradient^T * velocity >= -alpha * h
                constraint_value = np.dot(barrier_gradient, safe_velocity)
                required_value = -alpha * h
                
                if constraint_value < required_value:
                    # Project velocity onto safe set
                    correction = (required_value - constraint_value) * barrier_gradient
                    safe_velocity += correction
        
        return safe_velocity
    
    def move(self, control_input: Optional[np.ndarray] = None) -> bool:
        """
        Move the robot with safety-critical control.
        
        Args:
            control_input: Optional control input (default: move towards goal)
            
        Returns:
            True if movement was successful, False if emergency stop
        """
        # Check current safety status
        self.safety_status = self.check_safety_constraints()
        
        if self.safety_status == SafetyStatus.EMERGENCY_STOP:
            self.apply_emergency_stop()
            return False
        
        # Compute desired velocity
        if control_input is not None:
            desired_velocity = control_input
        else:
            # Simple goal-seeking behavior
            direction = self.goal_position - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                desired_velocity = direction / distance * min(1.0, distance)
            else:
                desired_velocity = np.zeros_like(self.position)
        
        # Apply safety-critical control
        safe_velocity = self.compute_safe_control(desired_velocity)
        
        # Update robot state
        dt = 0.1  # Time step
        self.acceleration = (safe_velocity - self.velocity) / dt
        self.velocity = safe_velocity
        self.position += self.velocity * dt
        self.timestamp += dt
        
        # Update metrics
        self.total_distance_traveled += np.linalg.norm(self.velocity) * dt
        
        # Store state history
        state = RobotState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            effort=self.effort.copy(),
            safety_status=self.safety_status,
            timestamp=self.timestamp
        )
        self.history.append(state)
        
        return True
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive safety metrics.
        
        Returns:
            Dictionary of safety metrics
        """
        if not self.history:
            return {}
        
        # Compute safety statistics
        safety_violations = sum(1 for state in self.history 
                              if state.safety_status in [SafetyStatus.CRITICAL, SafetyStatus.EMERGENCY_STOP])
        
        warning_count = sum(1 for state in self.history 
                           if state.safety_status == SafetyStatus.WARNING)
        
        # Compute trajectory smoothness
        velocities = np.array([state.velocity for state in self.history])
        accelerations = np.array([state.acceleration for state in self.history])
        
        jerk = np.linalg.norm(np.diff(accelerations, axis=0), axis=1)
        avg_jerk = np.mean(jerk) if len(jerk) > 0 else 0.0
        
        # Compute path efficiency
        total_path_length = np.sum(np.linalg.norm(np.diff(
            np.array([state.position for state in self.history]), axis=0
        ), axis=1)) if len(self.history) > 1 else 0.0
        
        straight_line_distance = np.linalg.norm(
            self.history[-1].position - self.history[0].position
        ) if len(self.history) > 1 else 0.0
        
        path_efficiency = straight_line_distance / total_path_length if total_path_length > 0 else 0.0
        
        return {
            "safety_violations": safety_violations,
            "warning_count": warning_count,
            "total_distance_traveled": self.total_distance_traveled,
            "path_efficiency": path_efficiency,
            "average_jerk": avg_jerk,
            "emergency_stops": 1 if self.emergency_stop_active else 0,
            "final_safety_status": self.safety_status.value,
            "simulation_time": self.timestamp
        }
    
    def reset(self, start_position: Optional[Tuple[float, float]] = None) -> None:
        """
        Reset the robot to initial state.
        
        Args:
            start_position: Optional new start position
        """
        if start_position is not None:
            self.position = np.array(start_position, dtype=float)
        
        self.velocity = np.zeros(2, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.effort = np.zeros(2, dtype=float)
        self.safety_status = SafetyStatus.SAFE
        self.emergency_stop_active = False
        self.timestamp = 0.0
        self.history = []
        self.safety_violations = 0
        self.total_distance_traveled = 0.0
        
        logger.info("Robot reset to initial state")
    
    def is_goal_reached(self, tolerance: float = 0.1) -> bool:
        """
        Check if the robot has reached the goal.
        
        Args:
            tolerance: Distance tolerance for goal reaching
            
        Returns:
            True if goal is reached
        """
        return np.linalg.norm(self.position - self.goal_position) < tolerance
