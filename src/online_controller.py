"""
Online Learning Controller

This module implements the online learning controller that adapts its
control policy in real-time using gradient-based optimization. The controller:

1. Uses an approximate model for feedforward control
2. Learns correction parameters online to compensate for model errors
3. Adapts to changing conditions and disturbances
4. Minimizes tracking error between reference and actual output

The learning algorithm follows the Online Gradient Descent (OGD) framework:
    θ(k+1) = θ(k) - η(k) * ∇L(θ(k))

where:
    θ - Controller parameters
    η - Learning rate (possibly adaptive)
    L - Loss/cost function (e.g., tracking error)

Reference:
    "Stochastic Online Optimization for Cyber-Physical and Robotic Systems"
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from .thermal_system import ApproximateThermalModel
from .gradient_estimator import (
    GradientEstimator, 
    FiniteDifferenceGradient,
    OnlineGradientEstimator
)


class OnlineLearningController:
    """
    Online learning controller for temperature regulation.
    
    Combines model-based feedforward control with online learning
    of correction parameters to handle model uncertainties and
    disturbances. Uses actual tracking error feedback to learn
    parameter corrections in real-time.
    """
    
    def __init__(
        self,
        approximate_model: ApproximateThermalModel,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        gradient_estimator: Optional[GradientEstimator] = None,
        control_bounds: Tuple[float, float] = (-10.0, 10.0)
    ):
        """
        Initialize the online learning controller.
        
        Args:
            approximate_model: Approximate system model for prediction
            learning_rate: Initial learning rate for online gradient descent
            momentum: Momentum coefficient for gradient updates
            gradient_estimator: Gradient estimation method (default: finite diff)
            control_bounds: (min, max) bounds for control input
        """
        self.model = approximate_model
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.control_bounds = control_bounds
        
        # Learnable parameters: [gain_correction, bias_correction, integral_gain]
        # gain_correction: multiplicative correction to control gain
        # bias_correction: additive correction to control bias
        # integral_gain: integral control gain for persistent errors
        self.parameters = np.array([1.0, 0.0, 0.1])
        
        # Integral term for accumulated error
        self.integral_error = 0.0
        
        # Previous state for gradient computation
        self.prev_error = 0.0
        self.prev_control = 0.0
        self.prev_base_control = 0.0
        
        # Momentum for parameter updates
        self.velocity = np.zeros(3)
        
        # Gradient estimator
        if gradient_estimator is None:
            base_estimator = FiniteDifferenceGradient(epsilon=0.01, method='central')
            self.gradient_estimator = OnlineGradientEstimator(
                base_estimator=base_estimator,
                momentum=0.0,  # We handle momentum separately
                gradient_clip=1.0
            )
        else:
            self.gradient_estimator = OnlineGradientEstimator(
                base_estimator=gradient_estimator,
                momentum=0.0,
                gradient_clip=1.0
            )
        
        # History tracking
        self.parameter_history = [self.parameters.copy()]
        self.error_history = []
        self.learning_rate_history = [learning_rate]
        self.gradient_history = []
        
        # State for online learning
        self.last_error = 0.0
        self.cumulative_error = 0.0
        self.step_count = 0
    
    def compute_control(
        self,
        current_temperature: float,
        reference_temperature: float,
        update_parameters: bool = True
    ) -> float:
        """
        Compute control action with online learning.
        
        Args:
            current_temperature: Current system temperature
            reference_temperature: Desired reference temperature
            update_parameters: Whether to update parameters (learning mode)
            
        Returns:
            Control input
        """
        # Tracking error
        error = reference_temperature - current_temperature
        self.last_error = error
        self.error_history.append(error)
        
        # Update integral error with anti-windup
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -50.0, 50.0)
        
        # Update parameters using actual error feedback (online learning)
        if update_parameters and self.step_count > 0:
            self._update_parameters_from_feedback(error)
        
        # Model-based feedforward control
        base_control = self.model.compute_control_for_target(
            current_temperature, reference_temperature
        )
        
        # Apply learned corrections
        # control = gain * base_control + bias + integral_gain * integral_error
        gain, bias, integral_gain = self.parameters
        corrected_control = gain * base_control + bias + integral_gain * self.integral_error
        
        # Bound control input
        bounded_control = np.clip(
            corrected_control, 
            self.control_bounds[0], 
            self.control_bounds[1]
        )
        
        # Store for next iteration
        self.prev_error = error
        self.prev_control = bounded_control
        self.prev_base_control = base_control
        
        self.step_count += 1
        
        return bounded_control
    
    def _update_parameters_from_feedback(self, current_error: float):
        """
        Update controller parameters using actual error feedback.
        
        This implements online gradient descent based on observed tracking error,
        not just model predictions. The gradient is estimated using the relationship
        between parameter changes and error reduction.
        
        Args:
            current_error: Current tracking error (reference - actual)
        """
        # Compute approximate gradients based on error sensitivity
        # For gain: if base_control and error have same sign, increase gain
        # For bias: directly proportional to persistent error
        # For integral_gain: proportional to accumulated error effect
        
        gain, bias, integral_gain = self.parameters
        
        # Error-based gradient approximation
        # dL/d(gain) ≈ -error * base_control (negative because we want to reduce error)
        # dL/d(bias) ≈ -error
        # dL/d(integral_gain) ≈ -error * integral_error
        
        gradient = np.array([
            -current_error * self.prev_base_control,
            -current_error,
            -current_error * self.integral_error * 0.01
        ])
        
        # Clip gradient for stability
        gradient = np.clip(gradient, -1.0, 1.0)
        
        self.gradient_history.append(gradient.copy())
        
        # Adaptive learning rate (decay over time for convergence)
        self.learning_rate = self.initial_learning_rate / (1 + 0.0005 * self.step_count)
        self.learning_rate_history.append(self.learning_rate)
        
        # Momentum update
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        
        # Parameter update with momentum
        self.parameters = self.parameters + self.velocity
        
        # Keep parameters in reasonable bounds
        self.parameters[0] = np.clip(self.parameters[0], 0.5, 2.0)  # gain: 0.5-2.0
        self.parameters[1] = np.clip(self.parameters[1], -5.0, 5.0)  # bias: -5 to 5
        self.parameters[2] = np.clip(self.parameters[2], 0.0, 0.5)  # integral_gain: 0-0.5
        
        # Record parameter history
        self.parameter_history.append(self.parameters.copy())
    
    def reset(self):
        """Reset controller state for new experiment."""
        self.parameters = np.array([1.0, 0.0, 0.1])
        self.parameter_history = [self.parameters.copy()]
        self.error_history = []
        self.learning_rate_history = [self.initial_learning_rate]
        self.gradient_history = []
        self.last_error = 0.0
        self.cumulative_error = 0.0
        self.step_count = 0
        self.learning_rate = self.initial_learning_rate
        self.gradient_estimator.reset()
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_control = 0.0
        self.prev_base_control = 0.0
        self.velocity = np.zeros(3)
    
    def get_parameters(self) -> np.ndarray:
        """Get current controller parameters."""
        return self.parameters.copy()
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get full history of controller states.
        
        Returns:
            Dictionary with parameter_history, error_history, etc.
        """
        return {
            'parameters': np.array(self.parameter_history),
            'errors': np.array(self.error_history),
            'learning_rates': np.array(self.learning_rate_history),
            'gradients': np.array(self.gradient_history) if self.gradient_history else np.array([])
        }


class StaticController:
    """
    Static (non-learning) controller for comparison.
    
    Uses only the approximate model without online adaptation.
    Demonstrates the limitation of static control under uncertainty.
    """
    
    def __init__(
        self,
        approximate_model: ApproximateThermalModel,
        control_bounds: Tuple[float, float] = (-10.0, 10.0)
    ):
        """
        Initialize static controller.
        
        Args:
            approximate_model: Approximate system model
            control_bounds: (min, max) bounds for control input
        """
        self.model = approximate_model
        self.control_bounds = control_bounds
        self.error_history = []
    
    def compute_control(
        self,
        current_temperature: float,
        reference_temperature: float
    ) -> float:
        """
        Compute control action without learning.
        
        Args:
            current_temperature: Current temperature
            reference_temperature: Reference temperature
            
        Returns:
            Control input
        """
        error = reference_temperature - current_temperature
        self.error_history.append(error)
        
        # Model-based feedforward only
        control = self.model.compute_control_for_target(
            current_temperature, reference_temperature
        )
        
        # Bound control
        bounded_control = np.clip(
            control, 
            self.control_bounds[0], 
            self.control_bounds[1]
        )
        
        return bounded_control
    
    def reset(self):
        """Reset controller state."""
        self.error_history = []
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get error history."""
        return {'errors': np.array(self.error_history)}


class PIDController:
    """
    Traditional PID controller for comparison.
    
    Uses proportional-integral-derivative control without
    model-based optimization or online learning.
    """
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        control_bounds: Tuple[float, float] = (-10.0, 10.0)
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            control_bounds: (min, max) bounds for control input
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.control_bounds = control_bounds
        
        # State
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
    
    def compute_control(
        self,
        current_temperature: float,
        reference_temperature: float
    ) -> float:
        """
        Compute PID control action.
        
        Args:
            current_temperature: Current temperature
            reference_temperature: Reference temperature
            
        Returns:
            Control input
        """
        error = reference_temperature - current_temperature
        self.error_history.append(error)
        
        # PID terms
        proportional = self.kp * error
        self.integral += self.ki * error
        derivative = self.kd * (error - self.last_error)
        
        self.last_error = error
        
        # Total control
        control = proportional + self.integral + derivative
        
        # Bound control with anti-windup
        bounded_control = np.clip(
            control,
            self.control_bounds[0],
            self.control_bounds[1]
        )
        
        # Anti-windup: limit integral if control is saturated
        if bounded_control != control:
            self.integral -= self.ki * error * 0.5
        
        return bounded_control
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get error history."""
        return {'errors': np.array(self.error_history)}
