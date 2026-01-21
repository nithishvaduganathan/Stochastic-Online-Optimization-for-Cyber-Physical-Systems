"""
Stochastic Disturbance Generator

This module implements various stochastic disturbance models that simulate
real-world uncertainties in cyber-physical systems, including:
- Gaussian noise (sensor noise, small fluctuations)
- Step disturbances (door opening/closing, sudden changes)
- Periodic disturbances (daily temperature cycles)
- Combined disturbances

These disturbances test the controller's robustness and adaptability.
"""

import numpy as np
from typing import Optional, List, Callable


class GaussianNoise:
    """
    Gaussian (white) noise generator.
    
    Simulates random fluctuations like sensor noise or
    small environmental variations.
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize Gaussian noise generator.
        
        Args:
            mean: Mean of the noise distribution
            std: Standard deviation of the noise
            seed: Random seed for reproducibility
        """
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(seed)
    
    def sample(self) -> float:
        """Generate a single noise sample."""
        return self.rng.normal(self.mean, self.std)
    
    def sample_sequence(self, n: int) -> np.ndarray:
        """Generate a sequence of n noise samples."""
        return self.rng.normal(self.mean, self.std, n)


class StepDisturbance:
    """
    Step disturbance generator.
    
    Simulates sudden changes like door opening, HVAC system
    switching, or sudden occupancy changes.
    """
    
    def __init__(
        self,
        step_time: int,
        step_magnitude: float,
        duration: Optional[int] = None
    ):
        """
        Initialize step disturbance.
        
        Args:
            step_time: Time step when the disturbance occurs
            step_magnitude: Magnitude of the step change
            duration: Duration of the step (None = permanent)
        """
        self.step_time = step_time
        self.step_magnitude = step_magnitude
        self.duration = duration
    
    def get_disturbance(self, time_step: int) -> float:
        """
        Get disturbance value at a given time step.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance value
        """
        if time_step >= self.step_time:
            if self.duration is None:
                return self.step_magnitude
            elif time_step < self.step_time + self.duration:
                return self.step_magnitude
        return 0.0


class PeriodicDisturbance:
    """
    Periodic disturbance generator.
    
    Simulates cyclic variations like daily temperature changes
    or regular occupancy patterns.
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        period: int = 100,
        phase: float = 0.0
    ):
        """
        Initialize periodic disturbance.
        
        Args:
            amplitude: Amplitude of the oscillation
            period: Period of the oscillation (in time steps)
            phase: Phase offset (in radians)
        """
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
    
    def get_disturbance(self, time_step: int) -> float:
        """
        Get disturbance value at a given time step.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance value
        """
        return self.amplitude * np.sin(
            2 * np.pi * time_step / self.period + self.phase
        )


class CombinedDisturbance:
    """
    Combined disturbance generator.
    
    Combines multiple disturbance sources for realistic scenarios.
    """
    
    def __init__(
        self,
        gaussian_noise: Optional[GaussianNoise] = None,
        step_disturbances: Optional[List[StepDisturbance]] = None,
        periodic_disturbances: Optional[List[PeriodicDisturbance]] = None
    ):
        """
        Initialize combined disturbance.
        
        Args:
            gaussian_noise: Gaussian noise component
            step_disturbances: List of step disturbances
            periodic_disturbances: List of periodic disturbances
        """
        self.gaussian_noise = gaussian_noise
        self.step_disturbances = step_disturbances or []
        self.periodic_disturbances = periodic_disturbances or []
    
    def get_disturbance(self, time_step: int) -> float:
        """
        Get combined disturbance value at a given time step.
        
        Args:
            time_step: Current time step
            
        Returns:
            Combined disturbance value
        """
        total = 0.0
        
        # Add Gaussian noise
        if self.gaussian_noise is not None:
            total += self.gaussian_noise.sample()
        
        # Add step disturbances
        for step in self.step_disturbances:
            total += step.get_disturbance(time_step)
        
        # Add periodic disturbances
        for periodic in self.periodic_disturbances:
            total += periodic.get_disturbance(time_step)
        
        return total


class EnvironmentChange:
    """
    Environment change simulator.
    
    Simulates sudden changes in environmental conditions,
    such as weather changes or seasonal transitions.
    """
    
    def __init__(
        self,
        change_time: int,
        new_ambient_temperature: float
    ):
        """
        Initialize environment change.
        
        Args:
            change_time: Time step when the change occurs
            new_ambient_temperature: New ambient temperature after change
        """
        self.change_time = change_time
        self.new_ambient_temperature = new_ambient_temperature
    
    def should_change(self, time_step: int) -> bool:
        """Check if environment should change at this time step."""
        return time_step == self.change_time


def create_realistic_disturbance(
    noise_std: float = 0.2,
    include_steps: bool = True,
    include_periodic: bool = True,
    seed: Optional[int] = None
) -> CombinedDisturbance:
    """
    Create a realistic combined disturbance scenario.
    
    Args:
        noise_std: Standard deviation of background noise
        include_steps: Include step disturbances
        include_periodic: Include periodic disturbances
        seed: Random seed for reproducibility
        
    Returns:
        CombinedDisturbance object
    """
    components = {}
    
    # Background noise
    components['gaussian_noise'] = GaussianNoise(mean=0.0, std=noise_std, seed=seed)
    
    # Step disturbances (e.g., door opening at t=200, window at t=500)
    if include_steps:
        components['step_disturbances'] = [
            StepDisturbance(step_time=200, step_magnitude=-0.5, duration=50),
            StepDisturbance(step_time=500, step_magnitude=0.8, duration=30),
        ]
    
    # Periodic disturbances (e.g., daily cycle)
    if include_periodic:
        components['periodic_disturbances'] = [
            PeriodicDisturbance(amplitude=0.3, period=200, phase=0.0),
        ]
    
    return CombinedDisturbance(**components)
