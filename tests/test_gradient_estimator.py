"""
Unit tests for gradient estimators.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gradient_estimator import (
    FiniteDifferenceGradient,
    SPSAGradient,
    OnlineGradientEstimator,
    create_gradient_estimator
)


class TestFiniteDifferenceGradient:
    """Tests for the FiniteDifferenceGradient class."""
    
    def test_forward_difference(self):
        """Test forward difference gradient estimation."""
        estimator = FiniteDifferenceGradient(epsilon=1e-6, method='forward')
        
        # Test on quadratic: f(x) = x^2, gradient = 2x
        def cost_func(params):
            return params[0] ** 2
        
        params = np.array([3.0])
        gradient = estimator.estimate(cost_func, params)
        
        # Expected gradient at x=3 is 6
        assert abs(gradient[0] - 6.0) < 1e-4
    
    def test_central_difference(self):
        """Test central difference gradient estimation."""
        estimator = FiniteDifferenceGradient(epsilon=1e-6, method='central')
        
        # Test on quadratic: f(x) = x^2, gradient = 2x
        def cost_func(params):
            return params[0] ** 2
        
        params = np.array([3.0])
        gradient = estimator.estimate(cost_func, params)
        
        # Central difference should be more accurate
        assert abs(gradient[0] - 6.0) < 1e-6
    
    def test_multidimensional(self):
        """Test gradient estimation for multiple parameters."""
        estimator = FiniteDifferenceGradient(epsilon=1e-6, method='central')
        
        # f(x, y) = x^2 + 2*y^2, gradient = [2x, 4y]
        def cost_func(params):
            return params[0] ** 2 + 2 * params[1] ** 2
        
        params = np.array([2.0, 3.0])
        gradient = estimator.estimate(cost_func, params)
        
        # Expected: [4.0, 12.0]
        assert abs(gradient[0] - 4.0) < 1e-5
        assert abs(gradient[1] - 12.0) < 1e-5
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            FiniteDifferenceGradient(method='invalid')


class TestSPSAGradient:
    """Tests for the SPSAGradient class."""
    
    def test_basic_estimation(self):
        """Test SPSA gradient estimation converges on average."""
        estimator = SPSAGradient(c=0.1, seed=42)
        
        # Test on quadratic: f(x) = x^2
        def cost_func(params):
            return params[0] ** 2
        
        params = np.array([3.0])
        
        # Average multiple estimates for accuracy
        gradients = []
        for i in range(100):
            estimator.rng = np.random.default_rng(i)
            gradient = estimator.estimate(cost_func, params)
            gradients.append(gradient[0])
        
        # Average should be close to true gradient (6.0)
        avg_gradient = np.mean(gradients)
        assert abs(avg_gradient - 6.0) < 0.5
    
    def test_multidimensional_efficiency(self):
        """Test SPSA uses only 2 function evaluations regardless of dimension."""
        estimator = SPSAGradient(c=0.1, seed=42)
        
        call_count = [0]
        
        def counting_cost_func(params):
            call_count[0] += 1
            return np.sum(params ** 2)
        
        # High dimensional case
        params = np.random.randn(100)
        _ = estimator.estimate(counting_cost_func, params)
        
        # Should only call function twice
        assert call_count[0] == 2


class TestOnlineGradientEstimator:
    """Tests for the OnlineGradientEstimator class."""
    
    def test_momentum(self):
        """Test momentum accumulation."""
        base = FiniteDifferenceGradient(epsilon=1e-4)
        estimator = OnlineGradientEstimator(
            base_estimator=base,
            momentum=0.9
        )
        
        def cost_func(params):
            return params[0] ** 2
        
        params = np.array([3.0])
        
        # First estimate
        grad1 = estimator.estimate(cost_func, params)
        
        # Second estimate (should have momentum)
        grad2 = estimator.estimate(cost_func, params)
        
        # With momentum, second gradient should be larger in magnitude
        assert abs(grad2[0]) > abs(grad1[0])
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        base = FiniteDifferenceGradient(epsilon=1e-4)
        estimator = OnlineGradientEstimator(
            base_estimator=base,
            gradient_clip=1.0
        )
        
        # Function with large gradient
        def cost_func(params):
            return 100 * params[0] ** 2
        
        params = np.array([3.0])
        gradient = estimator.estimate(cost_func, params)
        
        # Gradient should be clipped
        assert np.linalg.norm(gradient) <= 1.0 + 1e-10
    
    def test_reset(self):
        """Test momentum reset."""
        base = FiniteDifferenceGradient(epsilon=1e-4)
        estimator = OnlineGradientEstimator(
            base_estimator=base,
            momentum=0.9
        )
        
        def cost_func(params):
            return params[0] ** 2
        
        params = np.array([3.0])
        
        # Build up momentum
        for _ in range(5):
            estimator.estimate(cost_func, params)
        
        # Reset
        estimator.reset()
        
        # Velocity should be None
        assert estimator.velocity is None


class TestFactoryFunction:
    """Tests for the create_gradient_estimator factory."""
    
    def test_create_finite_difference(self):
        """Test creating finite difference estimator."""
        estimator = create_gradient_estimator('finite_difference', epsilon=0.01)
        assert isinstance(estimator, FiniteDifferenceGradient)
    
    def test_create_spsa(self):
        """Test creating SPSA estimator."""
        estimator = create_gradient_estimator('spsa', c=0.1)
        assert isinstance(estimator, SPSAGradient)
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            create_gradient_estimator('invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
