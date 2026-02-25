# Online Adaptive Cyber-Physical Control System Without Offline Pre-Training

## Abstract
This project presents an innovative Online Adaptive Cyber-Physical Control System (OACPS) designed to operate under stochastic disturbances with unknown system dynamics. Unlike traditional reinforcement learning methods that require extensive offline pre-training, our system utilizes a multi-agent architecture to learn and adapt a control policy in real-time using streaming sensor data. By implementing a gradient-based online optimization algorithm, the system continuously updates its control parameters to minimize error and adapt to changing environmental conditions. We compare the performance of this adaptive system against a traditional PID controller, demonstrating superior robustness and convergence in non-stationary environments.

## Problem Statement
Traditional control systems (like PID) require precise tuning and struggle with unknown or changing system dynamics. Modern AI-based controllers often require massive offline datasets and pre-training phases, which are impractical for systems where dynamics shift unpredictably (e.g., aging hardware, varying loads, or external disturbances). Additionally, fixed-gain controllers lack the self-tuning and fault-tolerance required for advanced final-year level applications.

## Phase 2 Enhancements
Beyond the core adaptive logic, this project implements:
1. **Real-time Web Dashboard**: Built with Flask and Chart.js for high-fidelity telemetry visualization.
2. **Meta-Adaptive Learning**: A higher-order agent that self-tunes the learning rate based on the loss trend to ensure stability and speed.
3. **Anomaly Detection Agent**: A cyber-security layer that monitors for sensor spoofing and rapid instability.
4. **Data Persistence**: A SQLite-based logging system for historical run analysis.

## Objectives
1. To develop a control system that operates without any offline pre-training.
2. To implement a multi-agent architecture for specialized task handling.
3. To adapt control policies in real-time using online gradient descent.
4. To provide both a high-fidelity Python simulation and a hardware implementation using ESP32.
5. To evaluate performance metrics (MSE, settling time, overshoot) against a baseline PID controller.

## System Architecture
The project is structured as a **Multi-Agent System (MAS)**:
- **Control Agent**: Handles the mathematical optimization and parameter updates.
- **Meta-Adaptive Agent**: Monitors the Control Agent's loss and tunes the learning rate.
- **Backend Agent**: Manages the Flask server, telemetry, and web dashboard.
- **Simulation Agent**: Models the stochastic environment and disturbances.
- **Security Agent**: Monitors for cyber-physical anomalies and sensor spoofing.
- **Hardware Agent**: Manages the cyber-physical interface (ESP32).
- **Evaluation Agent**: Analyzes performance and provides metrics.

## Mathematical Model & Algorithm
The system minimizes a cost function $J$ defined as:
$$J = \frac{1}{2}(y_{ref} - y_{actual})^2 + \frac{\lambda}{2}u^2$$
Where:
- $y_{ref}$ is the setpoint.
- $y_{actual}$ is the observed state.
- $u$ is the control effort.
- $\lambda$ is a regularization factor to prevent actuator saturation.

The control law is an adaptive parametric function $u = \theta^T \phi(x)$, where $\theta$ are weights and $\phi(x)$ are state features (error, integral, derivative). Weights are updated via:
$$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} J$$
Since the plant sensitivity $\frac{\partial y}{\partial u}$ is unknown, the system performs **Online Sensitivity Estimation** to approximate the gradient.

## Performance Analysis (Comparison with PID)
| Metric | PID Controller | Online Adaptive Agent |
| :--- | :--- | :--- |
| **Steady-State Error** | Low (if tuned) | Very Low (Self-adjusting) |
| **Overshoot** | Higher under disturbances| Minimized via adaptation |
| **Robustness** | Poor (Fixed gains) | High (Continuous learning) |
| **Setup Time** | Manual Tuning Required | Zero (Auto-adapting) |

## Innovation Highlights
- **Zero Pre-Training**: The system starts from scratch and stabilizes within seconds.
- **Cyber-Physical Continuity**: Seamless transition between simulated and physical control loops.
- **Gradient-Based Adaptation**: Mathematically rigorous approach to online learning.

## Future Scope
- Integration of Deep Neural Networks for more complex non-linear dynamics.
- Implementation of Decentralized Multi-Agent Control for distributed cyber-physical systems.
- Low-power optimization for edge deployment.

## Viva-Ready Explanation
"Our project solves the problem of controlling unknown systems by learning 'on-the-fly'. We used a Flask backend as a bridge between a simulated Python environment and a real ESP32 hardware. The core innovation is our Gradient-Based Online Optimization algorithm, which updates controller gains every 100ms based on the system's real-time response, effectively replacing the need for manual PID tuning or expensive pre-training."
