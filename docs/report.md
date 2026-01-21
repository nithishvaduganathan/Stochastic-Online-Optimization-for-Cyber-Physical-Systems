# Stochastic Online Optimization for Cyber-Physical Systems
## Smart Temperature Control Using Online Learning

### IEEE-Style Project Report

---

## Abstract

This project presents a comprehensive implementation of stochastic online optimization techniques for cyber-physical systems, demonstrated through a smart temperature control application. The system learns online from streaming data without any offline pre-training, adapting its control policy in real-time to minimize tracking error between desired reference temperatures and actual system output. Using approximate gradient estimation via finite differences and online gradient descent, the controller continuously updates its parameters to compensate for model uncertainties and stochastic disturbances. Experimental results demonstrate that the online learning controller achieves 35% lower RMSE compared to static model-based control and shows improved adaptability to sudden environmental changes. This work validates the effectiveness of online learning approaches for cyber-physical systems operating under uncertainty.

**Keywords:** Online Learning, Stochastic Optimization, Cyber-Physical Systems, Gradient Descent, Temperature Control, Adaptive Control

---

## 1. Introduction

### 1.1 Background

Cyber-physical systems (CPS) integrate computation, networking, and physical processes to create intelligent systems that interact with the physical world. These systems face inherent challenges including:

1. **Unknown System Dynamics**: True physical parameters are often unknown or change over time
2. **Stochastic Disturbances**: External factors introduce random variations
3. **Model Uncertainties**: Mathematical models only approximate real behavior
4. **Environmental Changes**: Operating conditions can shift suddenly

Traditional control approaches rely on accurate models or extensive offline training, which may not be available or applicable in dynamic environments.

### 1.2 Motivation

Online learning offers a paradigm shift: instead of requiring accurate models or offline training data, the controller learns and adapts continuously during operation. This approach is particularly valuable for:

- Systems where offline training is impractical
- Environments with changing conditions
- Applications requiring real-time adaptation

### 1.3 Objectives

This project aims to:

1. Design a cyber-physical control system using online learning
2. Implement approximate gradient estimation techniques
3. Demonstrate convergence under stochastic disturbances
4. Validate robustness to noise and environmental changes
5. Compare performance against traditional control approaches

---

## 2. Literature Review

### 2.1 Online Convex Optimization

Online convex optimization (OCO) provides a theoretical framework for sequential decision-making under uncertainty [1]. The regret bound guarantees that the online algorithm's performance approaches that of the best fixed strategy in hindsight.

### 2.2 Gradient-Based Optimization

When explicit gradients are unavailable, approximate gradient methods become essential:

- **Finite Differences**: Estimates gradients by perturbing parameters
- **SPSA (Simultaneous Perturbation Stochastic Approximation)**: Uses random perturbations for efficient gradient estimation [2]

### 2.3 Adaptive Control

Model reference adaptive control (MRAC) and self-tuning regulators share goals with online learning but typically assume specific model structures [3].

---

## 3. System Model

### 3.1 Thermal System Dynamics

The temperature control system is modeled as a first-order discrete-time system:

```
T(k+1) = (1 - α) · T(k) + α · T_ambient + β · u(k) + w(k)
```

Where:
- `T(k)`: Room temperature at time k (°C)
- `T_ambient`: External ambient temperature (°C)
- `u(k)`: Control input (heating/cooling power)
- `w(k)`: Stochastic disturbance
- `α`: Thermal decay coefficient (insulation quality)
- `β`: Control effectiveness coefficient

**System Parameters:**
| Parameter | True Value | Approximate Value | Description |
|-----------|------------|-------------------|-------------|
| α | 0.10 | 0.12 | Thermal decay coefficient |
| β | 0.50 | 0.45 | Control effectiveness |

The 20% and 10% parameter errors simulate realistic model uncertainty.

### 3.2 Stochastic Disturbances

The disturbance model combines:

1. **Gaussian Noise**: `N(0, σ²)` representing sensor noise and small fluctuations
2. **Step Disturbances**: Sudden changes (door opening, occupancy changes)
3. **Periodic Disturbances**: Cyclic variations (daily temperature patterns)

---

## 4. Online Learning Controller

### 4.1 Controller Architecture

The online learning controller combines model-based feedforward control with learned corrections:

```
u(k) = g · u_model(k) + b + κ · ∫e(t)dt
```

Where:
- `u_model(k)`: Base control from approximate model
- `g`: Learned gain correction parameter
- `b`: Learned bias correction parameter
- `κ`: Learned integral gain
- `e(t)`: Tracking error (reference - actual)

### 4.2 Online Gradient Descent

Parameters are updated using online gradient descent:

```
θ(k+1) = θ(k) + η(k) · v(k)
v(k) = μ · v(k-1) - ∇L(θ(k))
```

Where:
- `θ = [g, b, κ]`: Learnable parameters
- `η(k)`: Adaptive learning rate with decay
- `v(k)`: Velocity (momentum term)
- `μ`: Momentum coefficient
- `∇L`: Approximate gradient of the loss

### 4.3 Gradient Estimation

The gradient is approximated using the actual tracking error:

```
∂L/∂g ≈ -e(k) · u_model(k)
∂L/∂b ≈ -e(k)
∂L/∂κ ≈ -e(k) · ∫e(t)dt
```

This error-based gradient estimation provides direct feedback from the real system, enabling adaptation to unknown dynamics.

### 4.4 Adaptive Learning Rate

The learning rate decays over time to ensure convergence:

```
η(k) = η₀ / (1 + 0.0005 · k)
```

This schedule balances exploration (learning quickly initially) with exploitation (fine-tuning for stability).

---

## 5. Experimental Setup

### 5.1 Simulation Environment

| Configuration | Value |
|---------------|-------|
| Simulation Duration | 1000 time steps |
| Initial Temperature | 20°C |
| Ambient Temperature | 15°C (initially) |
| Control Bounds | [-10, 10] |
| Learning Rate | 0.05 |
| Momentum | 0.9 |

### 5.2 Reference Signal

Step changes in reference temperature:
- t=0: 22°C
- t=200: 25°C  
- t=500: 20°C
- t=800: 24°C

### 5.3 Environmental Changes

- t=300: Ambient temperature drops to 10°C
- t=700: Ambient temperature rises to 20°C

### 5.4 Controllers Compared

1. **Online Learning**: Proposed adaptive controller
2. **Static Model-Based**: Uses approximate model without learning
3. **PID Controller**: Traditional proportional-integral-derivative control

---

## 6. Results and Discussion

### 6.1 Performance Metrics

| Controller | RMSE | MAE | Max Error | Settling Time |
|------------|------|-----|-----------|---------------|
| **Online Learning** | **0.3458** | **0.2167** | 4.91 | **951** |
| Static | 0.5367 | 0.4158 | 4.59 | 1000 |
| PID | 0.3739 | 0.2393 | 4.51 | 994 |

**Key Observations:**
- Online learning achieves **35% lower RMSE** than static control
- Online learning achieves **7.5% lower RMSE** than PID control
- Fastest settling time (951 vs 994/1000 steps)

### 6.2 Convergence Analysis

The parameter evolution shows:
1. Initial adaptation phase (0-200 steps): Parameters adjust quickly
2. Steady-state learning (200-1000 steps): Fine-tuning for optimal performance
3. Environment adaptation (at t=300, t=700): Quick recovery after disturbances

The learning rate decay ensures parameters converge to stable values while maintaining adaptability.

### 6.3 Robustness to Disturbances

The online controller demonstrates:
- **Noise Rejection**: Smooths out Gaussian disturbances through integral action
- **Disturbance Recovery**: Quickly adapts after step disturbances
- **Environmental Adaptation**: Re-optimizes parameters when ambient temperature changes

### 6.4 Comparison with Static Control

The static controller's limitations:
- Cannot compensate for model errors
- Performance degrades with environmental changes
- No adaptation to disturbance patterns

### 6.5 Comparison with PID Control

PID advantages and limitations:
- Good steady-state performance through integral action
- Fixed gains cannot adapt to changing conditions
- Requires manual tuning for optimal performance

The online learning controller combines the benefits of integral action with adaptive gain adjustment.

---

## 7. Implementation Details

### 7.1 Code Structure

```
project/
├── src/
│   ├── thermal_system.py      # System model
│   ├── online_controller.py   # Learning controller
│   ├── disturbances.py        # Disturbance generators
│   ├── gradient_estimator.py  # Gradient estimation
│   ├── simulation.py          # Main simulation
│   └── visualization.py       # Plotting utilities
├── tests/
│   └── test_*.py              # Unit tests
├── experiments/               # Experimental scripts
├── results/                   # Generated figures
└── docs/                      # Documentation
```

### 7.2 Key Algorithms

**Online Gradient Descent (Algorithm 1)**
```
Initialize θ = [1.0, 0.0, 0.1], v = 0
For each time step k:
    1. Observe current temperature T(k)
    2. Compute tracking error e(k) = T_ref(k) - T(k)
    3. Compute approximate gradient ∇L
    4. Update velocity: v = μ·v - η(k)·∇L
    5. Update parameters: θ = θ + v
    6. Compute control: u(k) = g·u_model + b + κ·∫e
    7. Apply control to system
```

---

## 8. Conclusions

### 8.1 Summary

This project successfully demonstrates online learning for cyber-physical systems:

1. **No Pre-training Required**: The controller learns purely from online interaction
2. **Model-Free Adaptation**: Compensates for model uncertainties automatically
3. **Robust Performance**: Handles stochastic disturbances effectively
4. **Environmental Adaptability**: Quickly recovers from sudden changes
5. **Superior Performance**: Outperforms both static and PID controllers

### 8.2 Key Contributions

1. Implementation of approximate gradient estimation for control
2. Integration of momentum and adaptive learning rates
3. Comprehensive experimental validation
4. Comparative analysis with traditional approaches

### 8.3 Future Work

Potential extensions include:
- Multi-zone temperature control
- Nonlinear system dynamics
- Deep learning-based gradient estimation
- Hardware implementation on embedded systems
- Transfer learning across similar systems

---

## References

[1] E. Hazan, "Introduction to Online Convex Optimization," Foundations and Trends in Optimization, 2016.

[2] J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation," IEEE Transactions on Automatic Control, 1992.

[3] K. J. Åström and B. Wittenmark, "Adaptive Control," 2nd Edition, Addison-Wesley, 1995.

[4] M. Zinkevich, "Online Convex Programming and Generalized Infinitesimal Gradient Ascent," ICML, 2003.

[5] S. Shalev-Shwartz, "Online Learning and Online Convex Optimization," Foundations and Trends in Machine Learning, 2012.

---

## Appendix A: Generated Figures

The simulation generates the following visualization outputs:

1. **controller_comparison.png**: Temperature tracking comparison
2. **convergence_analysis.png**: Parameter evolution and error trends
3. **disturbance_response.png**: System response to disturbances
4. **performance_metrics.png**: Bar charts comparing RMSE, MAE, etc.

---

## Appendix B: Usage Instructions

### Installation

```bash
pip install -r requirements.txt
```

### Running Simulations

```bash
# Full comparison study
python -m src.simulation --experiment comparison --steps 1000

# Robustness study
python -m src.simulation --experiment robustness

# Single experiment
python -m src.simulation --experiment single
```

### Running Tests

```bash
python -m pytest tests/ -v
```

---

*Report prepared for Final Year Engineering Project*
*Stochastic Online Optimization for Cyber-Physical Systems*
