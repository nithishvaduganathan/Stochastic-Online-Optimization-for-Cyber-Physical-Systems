class PIDController:
    """
    Standard PID Controller for comparison with Adaptive Control.
    """
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute_control(self, y_ref, y_actual, dt=0.1):
        error = y_ref - y_actual
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        u = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        self.prev_error = error
        return u
