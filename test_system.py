#!/usr/bin/env python3
"""
Safety-Critical Control Systems - Quick Start Script

This script provides a quick way to test the safety-critical control system
and verify that all components are working correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits
        from safety_critical_control.controllers import (
            ControlBarrierFunctionController,
            ModelPredictiveControlController,
            EmergencyStopController,
            SafetyControllerManager
        )
        from safety_critical_control.evaluation import SafetyEvaluator
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of the safety-critical control system."""
    print("\nTesting basic functionality...")
    
    try:
        from safety_critical_control.robot import SafetyCriticalRobot, SafetyLimits
        from safety_critical_control.controllers import ControlBarrierFunctionController
        
        # Create robot
        safety_limits = SafetyLimits()
        robot = SafetyCriticalRobot(safety_limits=safety_limits, seed=42)
        
        # Create controller
        controller = ControlBarrierFunctionController()
        
        # Test basic movement
        state = np.concatenate([robot.position, robot.velocity])
        control, safety_info = controller.compute_control(
            state=state,
            reference=robot.goal_position,
            obstacles=robot.obstacles,
            safety_limits=safety_limits.__dict__
        )
        
        # Move robot
        success = robot.move(control)
        
        print(f"✓ Robot movement successful: {success}")
        print(f"✓ Robot position: {robot.position}")
        print(f"✓ Safety status: {robot.safety_status.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def test_evaluation():
    """Test evaluation system."""
    print("\nTesting evaluation system...")
    
    try:
        from safety_critical_control.evaluation import SafetyEvaluator
        
        evaluator = SafetyEvaluator()
        
        # Create dummy trajectory
        trajectory = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        obstacles = [np.array([1.5, 1.5])]
        goal = np.array([2.0, 2.0])
        control_history = [np.array([0.1, 0.1])] * 3
        safety_history = ['safe', 'safe', 'safe']
        timestamps = [0.0, 0.1, 0.2]
        
        metrics = evaluator.evaluate_trajectory(
            trajectory=trajectory,
            obstacles=obstacles,
            goal=goal,
            control_history=control_history,
            safety_history=safety_history,
            timestamps=timestamps,
            controller_name="Test",
            config={}
        )
        
        print(f"✓ Evaluation successful")
        print(f"✓ Success rate: {metrics.success_rate}")
        print(f"✓ Path efficiency: {metrics.path_efficiency}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Safety-Critical Control Systems - Quick Start Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: python 0680.py")
        print("2. Launch interactive demo: streamlit run demo/app.py")
        print("3. Run experiments: python scripts/run_simulation.py --controller CBF")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
