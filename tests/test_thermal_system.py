"""
Unit tests for the thermal system model.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.thermal_system import ThermalSystem, ApproximateThermalModel


class TestThermalSystem:
    """Tests for the ThermalSystem class."""
    
    def test_initialization(self):
        """Test system initializes with correct parameters."""
        system = ThermalSystem(
            alpha=0.1,
            beta=0.5,
            initial_temperature=20.0,
            ambient_temperature=15.0
        )
        
        assert system.alpha == 0.1
        assert system.beta == 0.5
        assert system.get_state() == 20.0
        assert system.ambient_temperature == 15.0
    
    def test_step_no_control_no_disturbance(self):
        """Test system decay towards ambient temperature."""
        system = ThermalSystem(
            alpha=0.1,
            beta=0.5,
            initial_temperature=25.0,
            ambient_temperature=15.0
        )
        
        # With no control or disturbance, temperature should decay
        for _ in range(100):
            temp = system.step(control_input=0.0, disturbance=0.0)
        
        # Should approach ambient temperature
        assert abs(temp - 15.0) < 0.1
    
    def test_step_with_control(self):
        """Test control input affects temperature."""
        system = ThermalSystem(
            alpha=0.1,
            beta=0.5,
            initial_temperature=20.0,
            ambient_temperature=15.0
        )
        
        # Apply heating
        temp = system.step(control_input=2.0, disturbance=0.0)
        
        # Temperature should increase due to heating
        expected = (1 - 0.1) * 20.0 + 0.1 * 15.0 + 0.5 * 2.0
        assert abs(temp - expected) < 1e-10
    
    def test_step_with_disturbance(self):
        """Test disturbance affects temperature."""
        system = ThermalSystem(
            alpha=0.1,
            beta=0.5,
            initial_temperature=20.0,
            ambient_temperature=15.0
        )
        
        # Apply disturbance
        temp = system.step(control_input=0.0, disturbance=1.0)
        
        expected = (1 - 0.1) * 20.0 + 0.1 * 15.0 + 1.0
        assert abs(temp - expected) < 1e-10
    
    def test_reset(self):
        """Test system reset functionality."""
        system = ThermalSystem(initial_temperature=20.0)
        
        # Run some steps
        for _ in range(10):
            system.step(1.0, 0.0)
        
        # Reset
        system.reset()
        
        assert system.get_state() == 20.0
        assert len(system.temperature_history) == 1
        assert len(system.control_history) == 0
    
    def test_set_ambient_temperature(self):
        """Test changing ambient temperature."""
        system = ThermalSystem(
            initial_temperature=20.0,
            ambient_temperature=15.0
        )
        
        system.set_ambient_temperature(25.0)
        assert system.ambient_temperature == 25.0
    
    def test_get_history(self):
        """Test history recording."""
        system = ThermalSystem(initial_temperature=20.0)
        
        # Run some steps
        for i in range(5):
            system.step(float(i), 0.1)
        
        temps, controls, disturbances = system.get_history()
        
        assert len(temps) == 6  # Initial + 5 steps
        assert len(controls) == 5
        assert len(disturbances) == 5
        assert all(d == 0.1 for d in disturbances)


class TestApproximateThermalModel:
    """Tests for the ApproximateThermalModel class."""
    
    def test_initialization(self):
        """Test model initializes correctly."""
        model = ApproximateThermalModel(
            estimated_alpha=0.12,
            estimated_beta=0.45,
            ambient_temperature=15.0
        )
        
        assert model.alpha == 0.12
        assert model.beta == 0.45
    
    def test_predict(self):
        """Test prediction functionality."""
        model = ApproximateThermalModel(
            estimated_alpha=0.1,
            estimated_beta=0.5,
            ambient_temperature=15.0
        )
        
        predicted = model.predict(
            current_temperature=20.0,
            control_input=2.0
        )
        
        expected = (1 - 0.1) * 20.0 + 0.1 * 15.0 + 0.5 * 2.0
        assert abs(predicted - expected) < 1e-10
    
    def test_compute_control_for_target(self):
        """Test control computation for target temperature."""
        model = ApproximateThermalModel(
            estimated_alpha=0.1,
            estimated_beta=0.5,
            ambient_temperature=15.0
        )
        
        # Compute control to maintain 20°C
        control = model.compute_control_for_target(
            current_temperature=20.0,
            target_temperature=20.0
        )
        
        # Verify: applying this control should result in target
        predicted = model.predict(20.0, control)
        assert abs(predicted - 20.0) < 1e-10
    
    def test_model_mismatch(self):
        """Test that approximate model differs from true system."""
        true_system = ThermalSystem(
            alpha=0.1,
            beta=0.5,
            initial_temperature=20.0
        )
        
        approx_model = ApproximateThermalModel(
            estimated_alpha=0.12,  # Different from true
            estimated_beta=0.45,   # Different from true
            ambient_temperature=15.0
        )
        
        control = 2.0
        
        # Get true next temperature
        true_next = true_system.step(control, 0.0)
        true_system.reset()
        
        # Get predicted next temperature
        predicted = approx_model.predict(20.0, control)
        
        # They should be different due to model mismatch
        assert abs(true_next - predicted) > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
