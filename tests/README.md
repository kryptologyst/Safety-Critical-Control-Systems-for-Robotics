# Safety-Critical Control Systems - Test Suite

This directory contains unit tests for the safety-critical control system components.

## Test Structure

- `test_robot.py`: Tests for SafetyCriticalRobot class
- `test_controllers.py`: Tests for safety controllers
- `test_evaluation.py`: Tests for evaluation metrics
- `test_integration.py`: Integration tests

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_robot.py

# Run with coverage
pytest --cov=safety_critical_control tests/

# Run with verbose output
pytest -v tests/
```
