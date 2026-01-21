"""
Unit tests for disturbance generators.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.disturbances import (
    GaussianNoise,
    StepDisturbance,
    PeriodicDisturbance,
    CombinedDisturbance,
    EnvironmentChange,
    create_realistic_disturbance
)


class TestGaussianNoise:
    """Tests for the GaussianNoise class."""
    
    def test_sample_statistics(self):
        """Test that samples have correct mean and std."""
        noise = GaussianNoise(mean=0.0, std=1.0, seed=42)
        
        samples = noise.sample_sequence(10000)
        
        assert abs(np.mean(samples) - 0.0) < 0.05
        assert abs(np.std(samples) - 1.0) < 0.05
    
    def test_reproducibility(self):
        """Test seed makes results reproducible."""
        noise1 = GaussianNoise(seed=42)
        noise2 = GaussianNoise(seed=42)
        
        samples1 = [noise1.sample() for _ in range(10)]
        samples2 = [noise2.sample() for _ in range(10)]
        
        assert samples1 == samples2


class TestStepDisturbance:
    """Tests for the StepDisturbance class."""
    
    def test_step_occurs_at_correct_time(self):
        """Test step disturbance activates at correct time."""
        step = StepDisturbance(step_time=100, step_magnitude=5.0)
        
        # Before step time
        assert step.get_disturbance(99) == 0.0
        
        # At and after step time
        assert step.get_disturbance(100) == 5.0
        assert step.get_disturbance(200) == 5.0
    
    def test_step_with_duration(self):
        """Test step disturbance with limited duration."""
        step = StepDisturbance(step_time=100, step_magnitude=5.0, duration=50)
        
        assert step.get_disturbance(99) == 0.0
        assert step.get_disturbance(100) == 5.0
        assert step.get_disturbance(149) == 5.0
        assert step.get_disturbance(150) == 0.0


class TestPeriodicDisturbance:
    """Tests for the PeriodicDisturbance class."""
    
    def test_amplitude(self):
        """Test periodic disturbance has correct amplitude."""
        periodic = PeriodicDisturbance(amplitude=2.0, period=100, phase=0.0)
        
        # Find max over one period
        values = [periodic.get_disturbance(t) for t in range(100)]
        
        assert abs(max(values) - 2.0) < 1e-10
        assert abs(min(values) + 2.0) < 1e-10
    
    def test_period(self):
        """Test periodic disturbance has correct period."""
        periodic = PeriodicDisturbance(amplitude=1.0, period=50, phase=0.0)
        
        val_0 = periodic.get_disturbance(0)
        val_50 = periodic.get_disturbance(50)
        val_100 = periodic.get_disturbance(100)
        
        # Values should repeat every 50 steps
        assert abs(val_0 - val_50) < 1e-10
        assert abs(val_0 - val_100) < 1e-10


class TestCombinedDisturbance:
    """Tests for the CombinedDisturbance class."""
    
    def test_combined_adds_components(self):
        """Test that combined disturbance sums all components."""
        step = StepDisturbance(step_time=0, step_magnitude=1.0)
        periodic = PeriodicDisturbance(amplitude=0.5, period=100, phase=np.pi/2)
        
        combined = CombinedDisturbance(
            step_disturbances=[step],
            periodic_disturbances=[periodic]
        )
        
        # At t=0: step=1.0, periodic=0.5*sin(pi/2)=0.5
        dist = combined.get_disturbance(0)
        assert abs(dist - 1.5) < 1e-10
    
    def test_empty_combined(self):
        """Test empty combined disturbance returns 0."""
        combined = CombinedDisturbance()
        
        assert combined.get_disturbance(50) == 0.0


class TestEnvironmentChange:
    """Tests for the EnvironmentChange class."""
    
    def test_change_at_correct_time(self):
        """Test environment change triggers at correct time."""
        change = EnvironmentChange(change_time=500, new_ambient_temperature=10.0)
        
        assert not change.should_change(499)
        assert change.should_change(500)
        assert not change.should_change(501)


class TestCreateRealisticDisturbance:
    """Tests for the create_realistic_disturbance function."""
    
    def test_creates_combined_disturbance(self):
        """Test factory creates combined disturbance."""
        disturbance = create_realistic_disturbance(
            noise_std=0.5,
            include_steps=True,
            include_periodic=True,
            seed=42
        )
        
        assert isinstance(disturbance, CombinedDisturbance)
        assert disturbance.gaussian_noise is not None
        assert len(disturbance.step_disturbances) > 0
        assert len(disturbance.periodic_disturbances) > 0
    
    def test_optional_components(self):
        """Test that components can be disabled."""
        disturbance = create_realistic_disturbance(
            include_steps=False,
            include_periodic=False
        )
        
        assert disturbance.gaussian_noise is not None
        assert len(disturbance.step_disturbances) == 0
        assert len(disturbance.periodic_disturbances) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
