import numpy as np

class PhysicalEnvironment:
    """
    Simulates a physical plant (e.g., a DC motor or a thermal system).
    Dynamics are intentionally complex or unknown to the controller.
    """
    def __init__(self, dt=0.1, noise_level=0.05):
        self.dt = dt
        self.noise_level = noise_level
        self.state = 0.0
        self.velocity = 0.0
        
        # System parameters (unknown to the controller)
        self.mass = 1.0
        self.damping = 0.5
        self.stiffness = 2.0

    def step(self, u, disturbance=0.0):
        """
        Second-order dynamics: m*x'' + c*x' + k*x = u + disturbance
        """
        acceleration = (u + disturbance - self.damping * self.velocity - self.stiffness * self.state) / self.mass
        
        # Update velocity and state (Euler integration)
        self.velocity += acceleration * self.dt
        self.state += self.velocity * self.dt
        
        # Add stochastic noise
        noise = np.random.normal(0, self.noise_level)
        observed_y = self.state + noise
        
        return observed_y

class DisturbanceGenerator:
    """
    Generates stochastic disturbances (e.g., step changes, gusts).
    """
    def __init__(self, probability=0.05, magnitude=1.0):
        self.probability = probability
        self.magnitude = magnitude

    def get_disturbance(self):
        if np.random.rand() < self.probability:
            return np.random.uniform(-self.magnitude, self.magnitude)
        return 0.0
