import os
import sys
from flask import Flask, request, jsonify, render_template

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.control_agent import ControlAgent
from core.security_agent import SecurityAgent
from backend.pid_controller import PIDController
import numpy as np

app = Flask(__name__, template_folder='templates')

# Initialize Controllers
adaptive_agent = ControlAgent(learning_rate=0.02, lambda_reg=0.05)
pid_agent = PIDController(Kp=1.5, Ki=0.5, Kd=0.1)
security_agent = SecurityAgent()

# Storage for history (telemetry)
history = {
    "adaptive": [],
    "pid": [],
    "reference": [],
    "u_adaptive": [],
    "u_pid": []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    """
    Receives current state from simulation or hardware.
    Returns control signals for both Adaptive and PID controllers.
    """
    data = request.json
    y_ref = data.get('y_ref', 0)
    y_actual = data.get('y_actual', 0)
    dt = data.get('dt', 0.1)
    
    # 1. Cyber-Security Check
    is_anomaly, message = security_agent.check_anomaly(y_actual)
    if is_anomaly:
        print(f"SECURITY ALERT: {message}")
    
    # 2. Adaptive Control logic
    u_adaptive = adaptive_agent.compute_control(y_ref, y_actual, dt)
    
    # Mitigation if anomalous
    u_adaptive = security_agent.apply_mitigation(u_adaptive, y_actual)
    
    # 3. PID Control logic (for comparison)
    u_pid = pid_agent.compute_control(y_ref, y_actual, dt)
    
    # Store history
    history["adaptive"].append(y_actual)
    history["pid"].append(data.get('y_pid', y_actual)) # Assume y_pid is passed for tracking if used
    history["reference"].append(y_ref)
    history["u_adaptive"].append(u_adaptive)
    history["u_pid"].append(u_pid)
    
    return jsonify({
        "u_adaptive": u_adaptive,
        "u_pid": u_pid
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Update the adaptive agent's parameters based on the new observation.
    """
    data = request.json
    y_ref = data.get('y_ref', 0)
    y_new = data.get('y_actual_new', 0)
    du = data.get('du', 0)
    dy = data.get('dy', 0)
    dt = data.get('dt', 0.1)
    
    # Update sensitivity first
    adaptive_agent.update_sensitivity(du, dy)
    
    # Update weights using gradient descent
    new_weights = adaptive_agent.update_parameters(y_ref, y_new, dt)
    
    return jsonify({
        "status": "updated",
        "weights": new_weights.tolist(),
        "sensitivity": adaptive_agent.sensitivity_estimate
    })

@app.route('/analytics', methods=['GET'])
def get_analytics():
    return jsonify(history)

if __name__ == '__main__':
    app.run(port=5000)
