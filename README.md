# Stochastic Online Optimization for Cyber-Physical Systems

A comprehensive implementation of online learning for cyber-physical control systems, demonstrated through a smart temperature control application.

## Overview

This project implements a cyber-physical control system that learns online from streaming data without any offline pre-training. The system operates under stochastic disturbances and unknown system dynamics, using an approximate model and gradient-based online optimization to continuously adapt its control policy.

### Key Features

- **Online Learning**: No pre-training required; learns in real-time from system feedback
- **Gradient-Based Optimization**: Uses approximate gradient estimation via finite differences
- **Stochastic Robustness**: Handles Gaussian noise, step disturbances, and periodic variations
- **Adaptive Control**: Automatically compensates for model uncertainties
- **Environmental Adaptation**: Quickly recovers from sudden environmental changes

## Project Structure

```
.
├── src/
│   ├── __init__.py            # Package initialization
│   ├── thermal_system.py      # Thermal system model
│   ├── online_controller.py   # Online learning controller
│   ├── disturbances.py        # Stochastic disturbance generators
│   ├── gradient_estimator.py  # Gradient estimation methods
│   ├── simulation.py          # Main simulation runner
│   └── visualization.py       # Plotting utilities
├── tests/
│   ├── test_thermal_system.py
│   ├── test_online_controller.py
│   ├── test_disturbances.py
│   └── test_gradient_estimator.py
├── docs/
│   └── report.md              # IEEE-style project report
├── results/                   # Generated figures
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Stochastic-Online-Optimization-for-Cyber-Physical-Systems

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Simulations

```bash
# Full comparison study (Online Learning vs Static vs PID)
python -m src.simulation --experiment comparison --steps 1000 --output results

# Robustness study under different noise levels
python -m src.simulation --experiment robustness

# Single experiment with online controller
python -m src.simulation --experiment single
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_online_controller.py -v
```

## Results

### Performance Comparison

| Controller | RMSE | MAE | Max Error | Settling Time |
|------------|------|-----|-----------|---------------|
| **Online Learning** | **0.3458** | **0.2167** | 4.91 | **951** |
| Static | 0.5367 | 0.4158 | 4.59 | 1000 |
| PID | 0.3739 | 0.2393 | 4.51 | 994 |

The online learning controller achieves:
- **35% lower RMSE** compared to static control
- **7.5% lower RMSE** compared to PID control
- Fastest settling time

### Generated Figures

The simulation generates the following visualizations in the `results/` directory:

1. `controller_comparison.png` - Temperature tracking comparison
2. `convergence_analysis.png` - Parameter evolution and error trends
3. `disturbance_response.png` - System response to disturbances
4. `performance_metrics.png` - Performance metrics comparison

## Theory

### Online Gradient Descent

The controller uses online gradient descent to adapt parameters in real-time:

```
θ(k+1) = θ(k) + η(k) · v(k)
v(k) = μ · v(k-1) - ∇L(θ(k))
```

Where:
- `θ = [gain, bias, integral_gain]`: Learnable parameters
- `η(k)`: Adaptive learning rate with decay
- `v(k)`: Velocity (momentum term)
- `∇L`: Approximate gradient estimated from tracking error

### Thermal System Model

```
T(k+1) = (1 - α) · T(k) + α · T_ambient + β · u(k) + w(k)
```

Where:
- `T(k)`: Room temperature
- `α`: Thermal decay coefficient
- `β`: Control effectiveness
- `u(k)`: Control input
- `w(k)`: Stochastic disturbance

## Documentation

See [docs/report.md](docs/report.md) for the complete IEEE-style project report including:
- Detailed methodology
- Mathematical formulation
- Experimental results
- Analysis and conclusions

## References

1. E. Hazan, "Introduction to Online Convex Optimization," 2016
2. J. C. Spall, "SPSA Gradient Approximation," IEEE TAC, 1992
3. K. J. Åström and B. Wittenmark, "Adaptive Control," 1995

## License

This project is developed for educational purposes as a final-year engineering project.

## Author

Stochastic Online Optimization for Cyber-Physical Systems