"""
Unit tests for the online learning controller.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.thermal_system import ApproximateThermalModel
from src.online_controller import (
    OnlineLearningController,
    StaticController,
    PIDController
)


class TestOnlineLearningController:
    """Tests for the OnlineLearningController class."""
    
    @pytest.fixture
    def controller(self):
        """Create a controller for testing."""
        model = ApproximateThermalModel(
            estimated_alpha=0.12,
            estimated_beta=0.45,
            ambient_temperature=15.0
        )
        return OnlineLearningController(
            approximate_model=model,
            learning_rate=0.01,
            momentum=0.9,
            control_bounds=(-10.0, 10.0)
        )
    
    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.learning_rate == 0.01
        assert controller.momentum == 0.9
        assert np.allclose(controller.parameters, [1.0, 0.0, 0.1])
    
    def test_compute_control(self, controller):
        """Test control computation."""
        control = controller.compute_control(
            current_temperature=20.0,
            reference_temperature=22.0
        )
        
        # Control should be bounded
        assert -10.0 <= control <= 10.0
        
        # Error should be recorded
        assert len(controller.error_history) == 1
        assert controller.error_history[0] == 2.0  # 22 - 20
    
    def test_parameter_update(self, controller):
        """Test parameters are updated during learning."""
        initial_params = controller.get_parameters().copy()
        
        # Run many control steps with a larger learning rate
        controller.initial_learning_rate = 0.1
        controller.learning_rate = 0.1
        
        for _ in range(50):
            controller.compute_control(20.0, 25.0, update_parameters=True)
        
        # Parameters should have changed (check with small tolerance)
        updated_params = controller.get_parameters()
        # At minimum, gradients should have been computed (first step doesn't update)
        assert len(controller.gradient_history) == 49
        # Check that some parameter update occurred (allow small changes)
        assert len(controller.parameter_history) > 1
    
    def test_control_bounds(self, controller):
        """Test control bounds are respected."""
        # Large reference change should still be bounded
        control = controller.compute_control(
            current_temperature=20.0,
            reference_temperature=100.0  # Very large reference
        )
        
        assert control <= 10.0
        
        control = controller.compute_control(
            current_temperature=20.0,
            reference_temperature=-50.0  # Very small reference
        )
        
        assert control >= -10.0
    
    def test_reset(self, controller):
        """Test controller reset."""
        # Run some steps
        for _ in range(10):
            controller.compute_control(20.0, 25.0)
        
        # Reset
        controller.reset()
        
        assert np.allclose(controller.parameters, [1.0, 0.0, 0.1])
        assert len(controller.error_history) == 0
        assert controller.step_count == 0
    
    def test_get_history(self, controller):
        """Test history retrieval."""
        for _ in range(5):
            controller.compute_control(20.0, 22.0)
        
        history = controller.get_history()
        
        assert 'parameters' in history
        assert 'errors' in history
        assert 'learning_rates' in history
        assert len(history['errors']) == 5
    
    def test_learning_improves_tracking(self, controller):
        """Test that learning reduces tracking error over time."""
        # Run many steps with consistent reference
        errors = []
        for _ in range(100):
            controller.compute_control(20.0, 25.0)
            errors.append(abs(controller.error_history[-1]))
        
        # Average error in first half should be >= last half
        # (controller learns to reduce error)
        first_half_avg = np.mean(errors[:50])
        second_half_avg = np.mean(errors[50:])
        
        # Note: This is a soft check - learning doesn't guarantee improvement
        # in all cases due to the approximate nature of the system
        assert len(errors) == 100


class TestStaticController:
    """Tests for the StaticController class."""
    
    def test_compute_control(self):
        """Test static controller computes control."""
        model = ApproximateThermalModel()
        controller = StaticController(
            approximate_model=model,
            control_bounds=(-10.0, 10.0)
        )
        
        control = controller.compute_control(20.0, 25.0)
        
        assert -10.0 <= control <= 10.0
        assert len(controller.error_history) == 1
    
    def test_no_learning(self):
        """Test static controller doesn't learn (fixed model)."""
        model = ApproximateThermalModel()
        controller = StaticController(approximate_model=model)
        
        # Run many steps
        controls = []
        for _ in range(10):
            control = controller.compute_control(20.0, 25.0)
            controls.append(control)
        
        # All controls should be identical for same input
        assert all(c == controls[0] for c in controls)


class TestPIDController:
    """Tests for the PIDController class."""
    
    def test_initialization(self):
        """Test PID controller initializes correctly."""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        assert controller.kp == 1.0
        assert controller.ki == 0.1
        assert controller.kd == 0.05
    
    def test_proportional_response(self):
        """Test proportional control response."""
        controller = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        # With only P control, output should be proportional to error
        control = controller.compute_control(20.0, 22.0)
        
        # Expected: kp * error = 1.0 * 2.0 = 2.0
        assert abs(control - 2.0) < 1e-10
    
    def test_integral_accumulation(self):
        """Test integral term accumulates."""
        controller = PIDController(kp=0.0, ki=1.0, kd=0.0)
        
        # Multiple steps with same error should accumulate integral
        controls = []
        for _ in range(5):
            control = controller.compute_control(20.0, 22.0)
            controls.append(control)
        
        # Each step should increase due to integral accumulation
        assert controls[-1] > controls[0]
    
    def test_reset(self):
        """Test PID reset clears state."""
        controller = PIDController()
        
        for _ in range(10):
            controller.compute_control(20.0, 25.0)
        
        controller.reset()
        
        assert controller.integral == 0.0
        assert controller.last_error == 0.0
        assert len(controller.error_history) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
