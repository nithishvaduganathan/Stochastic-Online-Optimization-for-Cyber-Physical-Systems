import numpy as np

class EvaluationAgent:
    """
    Computes performance metrics for control systems.
    """
    def compute_mse(self, reference, actual):
        return np.mean((np.array(reference) - np.array(actual))**2)

    def compute_overshoot(self, reference, actual):
        ref_val = reference[-1]
        max_val = np.max(actual)
        if max_val > ref_val:
            return (max_val - ref_val) / ref_val * 100
        return 0.0

    def compute_settling_time(self, reference, actual, dt, threshold=0.02):
        """
        Time taken for the response to stay within a threshold of the final value.
        """
        ref_val = reference[-1]
        for i in range(len(actual)-1, 0, -1):
            if abs(actual[i] - ref_val) > threshold * ref_val:
                return i * dt
        return 0.0

    def generate_report(self, results, dt=0.1):
        metrics = {
            "Adaptive": {
                "MSE": self.compute_mse(results["y_ref"], results["y_adaptive"]),
                "Overshoot (%)": self.compute_overshoot(results["y_ref"], results["y_adaptive"]),
                "Settling Time (s)": self.compute_settling_time(results["y_ref"], results["y_adaptive"], dt)
            },
            "PID": {
                "MSE": self.compute_mse(results["y_ref"], results["y_pid"]),
                "Overshoot (%)": self.compute_overshoot(results["y_ref"], results["y_pid"]),
                "Settling Time (s)": self.compute_settling_time(results["y_ref"], results["y_pid"], dt)
            }
        }
        return metrics
