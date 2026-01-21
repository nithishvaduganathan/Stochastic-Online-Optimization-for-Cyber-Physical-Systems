"""
Thermal System Model for Smart Temperature Control

This module implements a first-order thermal system model that simulates
a room's temperature dynamics under the influence of:
- Heating/cooling control input
- External environmental disturbances
- Stochastic noise

The system follows the dynamics:
    T(k+1) = (1 - alpha) * T(k) + alpha * T_ambient + beta * u(k) + disturbance(k)

where:
    T(k) - Current room temperature
    T_ambient - Ambient/external temperature
    u(k) - Control input (heating/cooling power)
    alpha - Thermal decay coefficient (related to insulation)
    beta - Control effectiveness coefficient
    disturbance(k) - Stochastic disturbance (noise, door opening, etc.)
"""

import numpy as np
from typing import Optional, Tuple


class ThermalSystem:
    """
    First-order discrete-time thermal system model.
    
    Models a room's temperature dynamics with controllable heating/cooling
    and stochastic disturbances representing real-world uncertainties.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.5,
        initial_temperature: float = 20.0,
        ambient_temperature: float = 15.0,
        dt: float = 1.0
    ):
        """
        Initialize the thermal system.
        
        Args:
            alpha: Thermal decay coefficient (0 < alpha < 1)
            beta: Control effectiveness coefficient (> 0)
            initial_temperature: Initial room temperature (°C)
            ambient_temperature: External ambient temperature (°C)
            dt: Time step (seconds)
        """
        self.alpha = alpha
        self.beta = beta
        self.temperature = initial_temperature
        self.ambient_temperature = ambient_temperature
        self.dt = dt
        
        # State history for analysis
        self.temperature_history = [initial_temperature]
        self.control_history = []
        self.disturbance_history = []
        
    def step(
        self,
        control_input: float,
        disturbance: float = 0.0
    ) -> float:
        """
        Advance the system by one time step.
        
        Args:
            control_input: Heating/cooling power (positive = heating, negative = cooling)
            disturbance: External disturbance (noise, door opening, etc.)
            
        Returns:
            New temperature after the time step
        """
        # System dynamics: T_new = (1-alpha)*T + alpha*T_ambient + beta*u + disturbance
        self.temperature = (
            (1 - self.alpha) * self.temperature +
            self.alpha * self.ambient_temperature +
            self.beta * control_input +
            disturbance
        )
        
        # Record history
        self.temperature_history.append(self.temperature)
        self.control_history.append(control_input)
        self.disturbance_history.append(disturbance)
        
        return self.temperature
    
    def reset(self, initial_temperature: Optional[float] = None) -> float:
        """
        Reset the system to initial state.
        
        Args:
            initial_temperature: New initial temperature (uses original if None)
            
        Returns:
            Initial temperature after reset
        """
        if initial_temperature is not None:
            self.temperature = initial_temperature
        else:
            self.temperature = self.temperature_history[0]
            
        self.temperature_history = [self.temperature]
        self.control_history = []
        self.disturbance_history = []
        
        return self.temperature
    
    def get_state(self) -> float:
        """Get current temperature."""
        return self.temperature
    
    def set_ambient_temperature(self, ambient_temperature: float) -> None:
        """
        Change the ambient temperature (simulates environmental change).
        
        Args:
            ambient_temperature: New ambient temperature
        """
        self.ambient_temperature = ambient_temperature
    
    def get_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full history of the system.
        
        Returns:
            Tuple of (temperature_history, control_history, disturbance_history)
        """
        return (
            np.array(self.temperature_history),
            np.array(self.control_history),
            np.array(self.disturbance_history)
        )


class ApproximateThermalModel:
    """
    Approximate thermal system model used by the controller.
    
    This model has imperfect knowledge of the true system parameters,
    simulating the scenario where the controller must learn online
    because the true system dynamics are unknown.
    """
    
    def __init__(
        self,
        estimated_alpha: float = 0.12,  # Slightly off from true value
        estimated_beta: float = 0.45,   # Slightly off from true value
        ambient_temperature: float = 15.0
    ):
        """
        Initialize the approximate model.
        
        Args:
            estimated_alpha: Estimated thermal decay coefficient
            estimated_beta: Estimated control effectiveness coefficient
            ambient_temperature: Estimated ambient temperature
        """
        self.alpha = estimated_alpha
        self.beta = estimated_beta
        self.ambient_temperature = ambient_temperature
    
    def predict(
        self,
        current_temperature: float,
        control_input: float
    ) -> float:
        """
        Predict the next temperature based on the approximate model.
        
        Args:
            current_temperature: Current room temperature
            control_input: Proposed control input
            
        Returns:
            Predicted next temperature
        """
        predicted = (
            (1 - self.alpha) * current_temperature +
            self.alpha * self.ambient_temperature +
            self.beta * control_input
        )
        return predicted
    
    def compute_control_for_target(
        self,
        current_temperature: float,
        target_temperature: float
    ) -> float:
        """
        Compute the control input needed to reach a target temperature.
        
        This is the model-based feedforward control component.
        
        Args:
            current_temperature: Current room temperature
            target_temperature: Desired target temperature
            
        Returns:
            Required control input
        """
        # Solve: target = (1-alpha)*current + alpha*ambient + beta*u
        # u = (target - (1-alpha)*current - alpha*ambient) / beta
        required_control = (
            target_temperature -
            (1 - self.alpha) * current_temperature -
            self.alpha * self.ambient_temperature
        ) / self.beta
        
        return required_control
