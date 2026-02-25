import numpy as np

class SecurityAgent:
    """
    Monitors the Cyber-Physical system for anomalies.
    Detects sensor spoofing, instability, or hardware malfunctions.
    """
    def __init__(self, threshold_spike=5.0, window_size=10):
        self.threshold_spike = threshold_spike
        self.window_size = window_size
        self.history = []
        self.is_anomalous = False

    def check_anomaly(self, y_actual):
        """
        Detects sudden spikes or excessive noise.
        """
        if len(self.history) < self.window_size:
            self.history.append(y_actual)
            return False, "Insufficient data"

        # Calculate moving average and standard deviation
        avg = np.mean(self.history)
        std = np.std(self.history)
        
        # 1. Spike Detection
        if abs(y_actual - avg) > self.threshold_spike:
            self.is_anomalous = True
            return True, "Sudden sensor spike detected (Potential Spoofing)"

        # 2. Instability Detection (High variance)
        if std > 2.0: # Arbitrary threshold for instability
            self.is_anomalous = True
            return True, "System instability detected (Excessive Jitter)"

        # Update history
        self.history.append(y_actual)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        self.is_anomalous = False
        return False, "Normal"

    def apply_mitigation(self, u_adaptive, y_actual):
        """
        If an anomaly is detected, we might want to scale down control effort
        or switch to a safe-mode PID.
        """
        if self.is_anomalous:
            return u_adaptive * 0.5 # Reduce gain for safety
        return u_adaptive
