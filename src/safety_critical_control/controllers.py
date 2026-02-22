"""
Advanced Safety-Critical Controllers

This module implements various safety-critical control algorithms including:
- Control Barrier Functions (CBF)
- Model Predictive Control (MPC) with safety constraints
- Emergency stop systems
- Robust control methods
"""

import numpy as np
import casadi as ca
from typing import Tuple, List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for safety-critical controllers."""
    # CBF parameters
    cbf_alpha: float = 1.0
    cbf_gamma: float = 0.1
    
    # MPC parameters
    mpc_horizon: int = 10
    mpc_dt: float = 0.1
    mpc_max_iter: int = 100
    
    # Safety parameters
    safety_margin: float = 0.5
    emergency_stop_threshold: float = 0.2
    
    # Control limits
    max_velocity: float = 2.0
    max_acceleration: float = 5.0
    max_effort: float = 10.0


class SafetyController(ABC):
    """Abstract base class for safety-critical controllers."""
    
    @abstractmethod
    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute safe control input.
        
        Args:
            state: Current robot state [x, y, vx, vy]
            reference: Reference trajectory or goal
            obstacles: List of obstacle positions
            safety_limits: Safety constraint limits
            
        Returns:
            Tuple of (control_input, safety_info)
        """
        pass


class ControlBarrierFunctionController(SafetyController):
    """
    Control Barrier Function (CBF) based safety controller.
    
    Implements CBF-QP formulation for safety-critical control.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize CBF controller."""
        self.config = config or ControllerConfig()
        logger.info("CBF controller initialized")
    
    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute CBF-based safe control.
        
        Args:
            state: Current state [x, y, vx, vy]
            reference: Reference control or goal
            obstacles: List of obstacle positions
            safety_limits: Safety limits
            
        Returns:
            Safe control input and safety information
        """
        position = state[:2]
        velocity = state[2:]
        
        # Desired control (simple proportional controller)
        desired_control = self._compute_desired_control(position, velocity, reference)
        
        # Apply CBF constraints
        safe_control, safety_info = self._apply_cbf_constraints(
            state, desired_control, obstacles, safety_limits
        )
        
        return safe_control, safety_info
    
    def _compute_desired_control(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """Compute desired control input."""
        if len(reference) == 2:  # Goal position
            error = reference - position
            desired_velocity = error * 0.5  # Proportional gain
        else:  # Reference velocity
            desired_velocity = reference
        
        # Simple velocity controller
        velocity_error = desired_velocity - velocity
        control = velocity_error * 2.0  # Proportional gain
        
        return control
    
    def _apply_cbf_constraints(
        self,
        state: np.ndarray,
        desired_control: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply CBF constraints using QP formulation."""
        position = state[:2]
        velocity = state[2:]
        
        # Initialize safety info
        safety_info = {
            "cbf_violations": 0,
            "active_constraints": [],
            "safety_margin": float('inf')
        }
        
        safe_control = desired_control.copy()
        
        # Apply CBF constraint for each obstacle
        for i, obstacle in enumerate(obstacles):
            # Distance to obstacle
            distance = np.linalg.norm(position - obstacle)
            
            if distance < safety_limits.get("warning_radius", 3.0):
                # CBF: h(x) = distance - safety_radius
                h = distance - safety_limits.get("safety_radius", 1.0)
                
                if h > 0:  # Only apply constraint if in safe region
                    # Gradient of barrier function
                    barrier_gradient = (position - obstacle) / distance
                    
                    # CBF constraint: L_f h + L_g h * u >= -alpha * h
                    # For simple dynamics: x_dot = u (velocity control)
                    L_f_h = 0.0  # No drift term
                    L_g_h = barrier_gradient
                    
                    # Check if constraint is violated
                    constraint_value = np.dot(L_g_h, safe_control)
                    required_value = -self.config.cbf_alpha * h
                    
                    if constraint_value < required_value:
                        # Project control onto safe set
                        correction = (required_value - constraint_value) * L_g_h
                        safe_control += correction
                        
                        safety_info["cbf_violations"] += 1
                        safety_info["active_constraints"].append(i)
                
                # Update minimum safety margin
                safety_info["safety_margin"] = min(safety_info["safety_margin"], h)
        
        # Apply control limits
        control_magnitude = np.linalg.norm(safe_control)
        if control_magnitude > safety_limits.get("max_velocity", 2.0):
            safe_control = safe_control / control_magnitude * safety_limits.get("max_velocity", 2.0)
        
        return safe_control, safety_info


class ModelPredictiveControlController(SafetyController):
    """
    Model Predictive Control (MPC) with safety constraints.
    
    Implements MPC using CasADi for optimization with safety constraints.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize MPC controller."""
        self.config = config or ControllerConfig()
        self._setup_mpc_problem()
        logger.info("MPC controller initialized")
    
    def _setup_mpc_problem(self) -> None:
        """Set up the MPC optimization problem using CasADi."""
        # Problem dimensions
        nx = 4  # state dimension [x, y, vx, vy]
        nu = 2  # control dimension [ax, ay]
        N = self.config.mpc_horizon
        
        # Decision variables
        X = ca.MX.sym('X', nx, N + 1)  # state trajectory
        U = ca.MX.sym('U', nu, N)      # control trajectory
        
        # Parameters
        x0 = ca.MX.sym('x0', nx)       # initial state
        x_ref = ca.MX.sym('x_ref', nx) # reference state
        obstacles = ca.MX.sym('obstacles', 2, 10)  # up to 10 obstacles
        
        # Dynamics function
        def dynamics(x, u):
            """Simple double integrator dynamics."""
            return ca.vertcat(
                x[2] + u[0] * self.config.mpc_dt,  # x_dot
                x[3] + u[1] * self.config.mpc_dt,  # y_dot
                u[0],  # vx_dot
                u[1]   # vy_dot
            )
        
        # Objective function
        obj = 0
        
        # State tracking cost
        for k in range(N + 1):
            state_error = X[:, k] - x_ref
            obj += ca.dot(state_error, state_error)
        
        # Control cost
        for k in range(N):
            obj += 0.1 * ca.dot(U[:, k], U[:, k])
        
        # Safety constraints
        g = []
        lbg = []
        ubg = []
        
        # Initial condition constraint
        g.append(X[:, 0] - x0)
        lbg.append([0, 0, 0, 0])
        ubg.append([0, 0, 0, 0])
        
        # Dynamics constraints
        for k in range(N):
            x_next = dynamics(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)
            lbg.append([0, 0, 0, 0])
            ubg.append([0, 0, 0, 0])
        
        # Safety constraints (collision avoidance)
        for k in range(N + 1):
            for i in range(10):  # Check up to 10 obstacles
                # Distance constraint: ||pos - obstacle|| >= safety_margin
                pos = X[:2, k]
                obs = obstacles[:, i]
                distance_sq = ca.dot(pos - obs, pos - obs)
                
                # Only add constraint if obstacle is valid (not zero)
                obs_valid = ca.dot(obs, obs) > 1e-6
                safety_constraint = ca.if_else(
                    obs_valid,
                    distance_sq - self.config.safety_margin**2,
                    0
                )
                
                g.append(safety_constraint)
                lbg.append(0)
                ubg.append(ca.inf)
        
        # Control constraints
        for k in range(N):
            # Velocity constraints
            g.append(ca.dot(X[2:4, k], X[2:4, k]))  # ||v||^2
            lbg.append(0)
            ubg.append(self.config.max_velocity**2)
            
            # Acceleration constraints
            g.append(ca.dot(U[:, k], U[:, k]))  # ||a||^2
            lbg.append(0)
            ubg.append(self.config.max_acceleration**2)
        
        # Create optimization problem
        nlp = {
            'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
            'f': obj,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, x_ref, obstacles.reshape((-1, 1)))
        }
        
        # Create solver
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.nx = nx
        self.nu = nu
        self.N = N
    
    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute MPC-based safe control.
        
        Args:
            state: Current state [x, y, vx, vy]
            reference: Reference state or goal
            obstacles: List of obstacle positions
            safety_limits: Safety limits
            
        Returns:
            Safe control input and safety information
        """
        # Prepare reference state
        if len(reference) == 2:  # Goal position
            x_ref = np.array([reference[0], reference[1], 0, 0])
        else:  # Reference state
            x_ref = reference
        
        # Prepare obstacles (pad with zeros if less than 10)
        obs_array = np.zeros((2, 10))
        for i, obs in enumerate(obstacles[:10]):
            obs_array[:, i] = obs
        
        # Solve MPC problem
        try:
            sol = self.solver(
                x0=np.zeros(self.nx * (self.N + 1) + self.nu * self.N),
                p=np.concatenate([state, x_ref, obs_array.flatten()]),
                lbg=[-ca.inf] * len(self.solver.stats()['g']),
                ubg=[ca.inf] * len(self.solver.stats()['g'])
            )
            
            if self.solver.stats()['success']:
                # Extract first control input
                control = np.array(sol['x'][self.nx * (self.N + 1):self.nx * (self.N + 1) + self.nu])
                
                safety_info = {
                    "mpc_solved": True,
                    "cost": float(sol['f']),
                    "iterations": self.solver.stats()['iter_count']
                }
            else:
                # Fallback to emergency stop
                control = np.zeros(2)
                safety_info = {
                    "mpc_solved": False,
                    "cost": float('inf'),
                    "iterations": 0,
                    "emergency_stop": True
                }
                logger.warning("MPC solver failed, applying emergency stop")
        
        except Exception as e:
            logger.error(f"MPC solver error: {e}")
            control = np.zeros(2)
            safety_info = {
                "mpc_solved": False,
                "cost": float('inf'),
                "iterations": 0,
                "emergency_stop": True,
                "error": str(e)
            }
        
        return control, safety_info


class EmergencyStopController(SafetyController):
    """
    Emergency stop controller with watchdog functionality.
    
    Monitors safety status and applies emergency stop when needed.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize emergency stop controller."""
        self.config = config or ControllerConfig()
        self.emergency_stop_active = False
        self.safety_violation_count = 0
        self.last_safe_time = 0.0
        logger.info("Emergency stop controller initialized")
    
    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute emergency stop control.
        
        Args:
            state: Current state
            reference: Reference (ignored in emergency stop)
            obstacles: List of obstacles
            safety_limits: Safety limits
            
        Returns:
            Emergency stop control and safety information
        """
        position = state[:2]
        velocity = state[2:]
        
        # Check for emergency conditions
        emergency_conditions = self._check_emergency_conditions(
            position, velocity, obstacles, safety_limits
        )
        
        safety_info = {
            "emergency_stop_active": self.emergency_stop_active,
            "safety_violations": self.safety_violation_count,
            "emergency_conditions": emergency_conditions
        }
        
        if emergency_conditions["critical_distance"] or emergency_conditions["velocity_limit"]:
            self.emergency_stop_active = True
            self.safety_violation_count += 1
            
            # Apply maximum braking
            control = -velocity * 5.0  # Braking force proportional to velocity
            
            # Limit braking force
            control_magnitude = np.linalg.norm(control)
            if control_magnitude > safety_limits.get("max_acceleration", 5.0):
                control = control / control_magnitude * safety_limits.get("max_acceleration", 5.0)
            
            safety_info["emergency_stop_active"] = True
            logger.critical("Emergency stop activated!")
        
        else:
            # Gradually reduce emergency stop if conditions improve
            if self.emergency_stop_active:
                # Check if safe conditions are maintained for some time
                if self._is_safe_for_duration(1.0):  # 1 second of safety
                    self.emergency_stop_active = False
                    logger.info("Emergency stop deactivated")
            
            if self.emergency_stop_active:
                control = np.zeros(2)  # Stop
            else:
                control = np.zeros(2)  # No control (emergency stop controller)
        
        return control, safety_info
    
    def _check_emergency_conditions(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check for emergency conditions."""
        conditions = {
            "critical_distance": False,
            "velocity_limit": False,
            "boundary_violation": False
        }
        
        # Check critical distance to obstacles
        min_distance = float('inf')
        for obstacle in obstacles:
            distance = np.linalg.norm(position - obstacle)
            min_distance = min(min_distance, distance)
            
            if distance < self.config.emergency_stop_threshold:
                conditions["critical_distance"] = True
        
        # Check velocity limits
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > safety_limits.get("max_velocity", 2.0) * 1.2:  # 20% tolerance
            conditions["velocity_limit"] = True
        
        # Check boundary violations
        if np.any(position < 0) or np.any(position > 10):  # Assuming 10x10 grid
            conditions["boundary_violation"] = True
        
        return conditions
    
    def _is_safe_for_duration(self, duration: float) -> bool:
        """Check if safe conditions have been maintained for given duration."""
        # Simplified implementation - in practice would track timestamps
        return not self.emergency_stop_active
    
    def reset(self) -> None:
        """Reset emergency stop controller."""
        self.emergency_stop_active = False
        self.safety_violation_count = 0
        self.last_safe_time = 0.0
        logger.info("Emergency stop controller reset")


class SafetyControllerManager:
    """
    Manager for multiple safety controllers with priority handling.
    
    Combines different safety controllers with priority-based execution.
    """
    
    def __init__(self, controllers: List[SafetyController]):
        """
        Initialize controller manager.
        
        Args:
            controllers: List of safety controllers in priority order
        """
        self.controllers = controllers
        self.active_controller = 0
        logger.info(f"Safety controller manager initialized with {len(controllers)} controllers")
    
    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        obstacles: List[np.ndarray],
        safety_limits: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control using active controller.
        
        Args:
            state: Current state
            reference: Reference trajectory
            obstacles: List of obstacles
            safety_limits: Safety limits
            
        Returns:
            Control input and safety information
        """
        if not self.controllers:
            return np.zeros(2), {"error": "No controllers available"}
        
        # Use active controller
        controller = self.controllers[self.active_controller]
        control, safety_info = controller.compute_control(
            state, reference, obstacles, safety_limits
        )
        
        # Add controller info
        safety_info["active_controller"] = self.active_controller
        safety_info["controller_type"] = type(controller).__name__
        
        return control, safety_info
    
    def switch_controller(self, controller_index: int) -> bool:
        """
        Switch to a different controller.
        
        Args:
            controller_index: Index of controller to switch to
            
        Returns:
            True if switch was successful
        """
        if 0 <= controller_index < len(self.controllers):
            self.active_controller = controller_index
            logger.info(f"Switched to controller {controller_index}: {type(self.controllers[controller_index]).__name__}")
            return True
        return False
    
    def add_controller(self, controller: SafetyController) -> None:
        """Add a new controller to the manager."""
        self.controllers.append(controller)
        logger.info(f"Added controller: {type(controller).__name__}")
    
    def get_controller_info(self) -> List[Dict[str, Any]]:
        """Get information about all controllers."""
        return [
            {
                "index": i,
                "type": type(controller).__name__,
                "active": i == self.active_controller
            }
            for i, controller in enumerate(self.controllers)
        ]
