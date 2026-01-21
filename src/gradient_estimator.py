"""
Approximate Gradient Estimator

This module implements gradient estimation techniques for online optimization
when the true gradient is not available. The key methods include:

1. Finite Difference Gradient Estimation
   - Forward difference
   - Central difference
   - Simultaneous perturbation (SPSA)

2. Zeroth-Order Gradient Estimation
   - Based on function evaluations only

These methods are crucial for online learning in cyber-physical systems
where the true system model is unknown and gradients cannot be computed
analytically.

Reference:
    Spall, J. C. (1992). Multivariate stochastic approximation using a 
    simultaneous perturbation gradient approximation. IEEE Transactions 
    on Automatic Control.
"""

import numpy as np
from typing import Callable, Optional, Union
from abc import ABC, abstractmethod


class GradientEstimator(ABC):
    """Abstract base class for gradient estimators."""
    
    @abstractmethod
    def estimate(
        self,
        cost_function: Callable[[np.ndarray], float],
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the gradient of the cost function at given parameters.
        
        Args:
            cost_function: Function that returns cost given parameters
            parameters: Current parameter values
            
        Returns:
            Estimated gradient
        """
        pass


class FiniteDifferenceGradient(GradientEstimator):
    """
    Finite difference gradient estimator.
    
    Uses small perturbations to approximate gradients numerically.
    Supports both forward and central difference methods.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-4,
        method: str = 'central'
    ):
        """
        Initialize finite difference gradient estimator.
        
        Args:
            epsilon: Perturbation size
            method: 'forward' or 'central' difference
        """
        self.epsilon = epsilon
        self.method = method
        
        if method not in ['forward', 'central']:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate(
        self,
        cost_function: Callable[[np.ndarray], float],
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Estimate gradient using finite differences.
        
        Args:
            cost_function: Function that returns cost given parameters
            parameters: Current parameter values
            
        Returns:
            Estimated gradient
        """
        n_params = len(parameters)
        gradient = np.zeros(n_params)
        
        if self.method == 'forward':
            f_x = cost_function(parameters)
            for i in range(n_params):
                params_plus = parameters.copy()
                params_plus[i] += self.epsilon
                f_x_plus = cost_function(params_plus)
                gradient[i] = (f_x_plus - f_x) / self.epsilon
                
        else:  # central difference
            for i in range(n_params):
                params_plus = parameters.copy()
                params_minus = parameters.copy()
                params_plus[i] += self.epsilon
                params_minus[i] -= self.epsilon
                f_plus = cost_function(params_plus)
                f_minus = cost_function(params_minus)
                gradient[i] = (f_plus - f_minus) / (2 * self.epsilon)
        
        return gradient


class SPSAGradient(GradientEstimator):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) gradient estimator.
    
    Uses random simultaneous perturbations for efficient gradient estimation.
    Requires only 2 function evaluations regardless of parameter dimension,
    making it highly efficient for high-dimensional problems.
    
    Reference:
        Spall, J.C. (1992). IEEE Trans. Automatic Control.
    """
    
    def __init__(
        self,
        c: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize SPSA gradient estimator.
        
        Args:
            c: Perturbation magnitude parameter
            seed: Random seed for reproducibility
        """
        self.c = c
        self.rng = np.random.default_rng(seed)
    
    def estimate(
        self,
        cost_function: Callable[[np.ndarray], float],
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Estimate gradient using SPSA.
        
        Args:
            cost_function: Function that returns cost given parameters
            parameters: Current parameter values
            
        Returns:
            Estimated gradient
        """
        n_params = len(parameters)
        
        # Generate random perturbation direction (Bernoulli ±1)
        delta = self.rng.choice([-1, 1], size=n_params).astype(float)
        
        # Perturbed parameters
        params_plus = parameters + self.c * delta
        params_minus = parameters - self.c * delta
        
        # Function evaluations
        f_plus = cost_function(params_plus)
        f_minus = cost_function(params_minus)
        
        # SPSA gradient estimate
        gradient = (f_plus - f_minus) / (2 * self.c * delta)
        
        return gradient


class OnlineGradientEstimator:
    """
    Online gradient estimator for real-time control applications.
    
    Combines gradient estimation with momentum and adaptive learning
    for stable online optimization.
    """
    
    def __init__(
        self,
        base_estimator: GradientEstimator,
        momentum: float = 0.0,
        gradient_clip: Optional[float] = None
    ):
        """
        Initialize online gradient estimator.
        
        Args:
            base_estimator: Underlying gradient estimator
            momentum: Momentum coefficient (0 = no momentum)
            gradient_clip: Maximum gradient magnitude (None = no clipping)
        """
        self.base_estimator = base_estimator
        self.momentum = momentum
        self.gradient_clip = gradient_clip
        self.velocity = None
    
    def estimate(
        self,
        cost_function: Callable[[np.ndarray], float],
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Estimate gradient with momentum and clipping.
        
        Args:
            cost_function: Function that returns cost given parameters
            parameters: Current parameter values
            
        Returns:
            Processed gradient estimate
        """
        # Get base gradient estimate
        gradient = self.base_estimator.estimate(cost_function, parameters)
        
        # Apply gradient clipping
        if self.gradient_clip is not None:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > self.gradient_clip:
                gradient = gradient * (self.gradient_clip / grad_norm)
        
        # Apply momentum
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(gradient)
            self.velocity = self.momentum * self.velocity + gradient
            return self.velocity
        
        return gradient
    
    def reset(self):
        """Reset the momentum velocity."""
        self.velocity = None


def create_gradient_estimator(
    method: str = 'finite_difference',
    **kwargs
) -> GradientEstimator:
    """
    Factory function to create gradient estimators.
    
    Args:
        method: 'finite_difference' or 'spsa'
        **kwargs: Additional arguments for the specific estimator
        
    Returns:
        GradientEstimator instance
    """
    if method == 'finite_difference':
        return FiniteDifferenceGradient(**kwargs)
    elif method == 'spsa':
        return SPSAGradient(**kwargs)
    else:
        raise ValueError(f"Unknown gradient estimation method: {method}")
