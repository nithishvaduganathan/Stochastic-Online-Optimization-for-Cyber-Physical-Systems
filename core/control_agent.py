import numpy as np

class ControlAgent:
    """
    Online Adaptive Control Agent using Gradient-Based Optimization.
    Learns continuously from streaming data without offline pre-training.
    """
    def __init__(self, learning_rate=0.01, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        
        # Meta-Adaptive Parameters
        self.loss_history = []
        self.min_lr = 0.001
        self.max_lr = 0.05
        
        # Adaptive parameters: u = w1 * error + w2 * integral_error + w3 * derivative_error
        # This is essentially an adaptive PID where the gains are updated online.
        self.weights = np.array([0.1, 0.01, 0.05]) 
        
        self.prev_error = 0
        self.integral_error = 0
        
        # Estimated plant sensitivity (dy/du) - updated online
        # Since dynamics are unknown, we start with a positive guess (assuming positive gain)
        self.sensitivity_estimate = 1.0 
        self.sensitivity_lr = 0.05

    def compute_control(self, y_ref, y_actual, dt=0.1):
        error = y_ref - y_actual
        self.integral_error += error * dt
        derivative_error = (error - self.prev_error) / dt
        
        features = np.array([error, self.integral_error, derivative_error])
        
        # Linear control law: u = w^T * features
        u = np.dot(self.weights, features)
        
        # Clipping control signal for safety
        u = np.clip(u, -100, 100)
        
        self.prev_error = error
        self.last_features = features
        self.last_u = u
        
        return u

    def update_parameters(self, y_ref, y_actual_new, dt=0.1):
        """
        Update weights using gradient descent:
        Loss J = 0.5 * (y_ref - y_actual)^2 + 0.5 * lambda * u^2
        grad_w = (y_actual - y_ref) * (dy/du) * (du/dw) + lambda * u * (du/dw)
        du/dw = features
        """
        error = y_ref - y_actual_new
        
        # 1. Update sensitivity estimate (Online System Identification)
        # dy = sensitivity * du
        # We use a simple delta-rule to update the estimate of dy/du
        # This is a simplification; in real systems, we'd use recursive least squares or similar.
        # But for "gradient-based online optimization", we can use this.
        # Note: This requires the previous u and the change in y.
        # (Actually, sensitivity is better updated in a separate step or inferred)
        
        # 2. Update control weights
        # Gradient of loss J w.r.t weights w:
        # dJ/dw = -(y_ref - y_actual) * sensitivity * features + lambda * u * features
        
        grad_w = (-error * self.sensitivity_estimate * self.last_features + 
                  self.lambda_reg * self.last_u * self.last_features)
        
        # Meta-Adaptive Logic: Adjust learning rate based on loss trend
        current_loss = 0.5 * error**2
        if len(self.loss_history) > 0:
            if current_loss < self.loss_history[-1]:
                # Loss is decreasing, we can slightly increase LR to speed up convergence
                self.learning_rate = min(self.max_lr, self.learning_rate * 1.05)
            else:
                # Loss is increasing or oscillating, reduce LR for stability
                self.learning_rate = max(self.min_lr, self.learning_rate * 0.7)
        
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 50: self.loss_history.pop(0)

        # Gradient Descent Step
        self.weights -= self.learning_rate * grad_w
        
        return self.weights

    def update_sensitivity(self, du, dy):
        """
        Estimates the plant's response to control input (sensitivity).
        dy = sensitivity * du -> sensitivity = dy/du
        """
        if abs(du) > 1e-5:
            instant_sensitivity = dy / du
            # Smooth the estimate
            self.sensitivity_estimate += self.sensitivity_lr * (instant_sensitivity - self.sensitivity_estimate)
            # Ensure it doesn't flip sign unexpectedly or go to zero if we know plant is generally positive
            self.sensitivity_estimate = np.clip(self.sensitivity_estimate, 0.1, 10.0)
