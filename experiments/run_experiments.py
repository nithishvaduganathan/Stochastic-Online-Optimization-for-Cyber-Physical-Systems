"""
Experimental Evaluation Scripts

This module contains additional experiments for evaluating the
online learning controller under various conditions.
"""

import os
import sys

# Add parent directory to path before importing other modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.simulation import run_experiment, run_comparison_study, run_robustness_study
from src.thermal_system import ThermalSystem, ApproximateThermalModel
from src.online_controller import OnlineLearningController
from src.simulation import generate_reference_signal
from src.disturbances import create_realistic_disturbance, GaussianNoise


def run_learning_rate_sensitivity():
    """Study the effect of different learning rates."""
    print("\n" + "=" * 70)
    print("LEARNING RATE SENSITIVITY STUDY")
    print("=" * 70)
    
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        np.random.seed(42)
        n_steps = 500
        
        system = ThermalSystem(
            alpha=0.1, beta=0.5,
            initial_temperature=20.0,
            ambient_temperature=15.0
        )
        
        model = ApproximateThermalModel(
            estimated_alpha=0.12,
            estimated_beta=0.45
        )
        
        controller = OnlineLearningController(
            approximate_model=model,
            learning_rate=lr,
            momentum=0.9
        )
        
        reference = generate_reference_signal(n_steps, 'step')
        disturbance = create_realistic_disturbance(noise_std=0.2, seed=42)
        
        errors = []
        for k in range(n_steps):
            current_temp = system.get_state()
            ref_temp = reference[k]
            control = controller.compute_control(current_temp, ref_temp)
            dist = disturbance.get_disturbance(k)
            system.step(control, dist)
            errors.append(ref_temp - system.get_state())
        
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        results[lr] = {
            'rmse': rmse,
            'errors': np.array(errors),
            'parameters': np.array(controller.parameter_history)
        }
        print(f"  RMSE: {rmse:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    lrs = list(results.keys())
    rmses = [results[lr]['rmse'] for lr in lrs]
    
    axes[0].plot(lrs, rmses, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE vs Learning Rate')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lrs)))
    for idx, lr in enumerate(lrs):
        axes[1].plot(np.abs(results[lr]['errors']), 
                    color=colors[idx], alpha=0.7, label=f'η={lr}')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Error Evolution for Different Learning Rates')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/learning_rate_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/learning_rate_sensitivity.png")
    
    return results


def run_noise_adaptation_demo():
    """Demonstrate adaptation to changing noise levels."""
    print("\n" + "=" * 70)
    print("NOISE ADAPTATION DEMONSTRATION")
    print("=" * 70)
    
    np.random.seed(42)
    n_steps = 1500
    
    # Noise levels change during simulation
    noise_schedule = [
        (0, 0.1),      # Low noise initially
        (500, 0.5),    # High noise
        (1000, 0.2),   # Medium noise
    ]
    
    system = ThermalSystem(alpha=0.1, beta=0.5, initial_temperature=20.0)
    model = ApproximateThermalModel()
    controller = OnlineLearningController(approximate_model=model, learning_rate=0.05)
    
    reference = np.ones(n_steps) * 22.0
    
    temperatures = []
    current_noise_std = 0.1
    noise_gen = GaussianNoise(std=current_noise_std, seed=42)
    
    for k in range(n_steps):
        # Check for noise level changes
        for change_time, new_std in noise_schedule:
            if k == change_time:
                current_noise_std = new_std
                noise_gen = GaussianNoise(std=new_std, seed=42 + k)
                print(f"Time {k}: Noise level changed to σ={new_std}")
        
        current_temp = system.get_state()
        control = controller.compute_control(current_temp, reference[k])
        dist = noise_gen.sample()
        system.step(control, dist)
        temperatures.append(system.get_state())
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time = np.arange(n_steps)
    axes[0].plot(time, reference, 'b--', label='Reference', linewidth=2)
    axes[0].plot(time, temperatures, 'r-', label='Actual', alpha=0.8)
    
    # Mark noise changes
    for change_time, new_std in noise_schedule:
        axes[0].axvline(x=change_time, color='orange', linestyle=':', linewidth=2)
        axes[0].annotate(f'σ={new_std}', xy=(change_time, 23.5), fontsize=10)
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Adaptation to Changing Noise Levels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    errors = reference - np.array(temperatures)
    window = 50
    moving_avg = np.convolve(np.abs(errors), np.ones(window)/window, mode='valid')
    axes[1].plot(time, np.abs(errors), 'b-', alpha=0.3, label='|Error|')
    axes[1].plot(time[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Avg')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('|Error| (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/noise_adaptation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/noise_adaptation.png")


def run_all_experiments():
    """Run all experimental studies."""
    os.makedirs('results', exist_ok=True)
    
    # Main comparison study
    run_comparison_study(n_steps=1000, output_dir='results')
    
    # Robustness study
    run_robustness_study(n_trials=5, n_steps=500, output_dir='results')
    
    # Learning rate sensitivity
    run_learning_rate_sensitivity()
    
    # Noise adaptation
    run_noise_adaptation_demo()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("Results saved to 'results/' directory")
    print("=" * 70)


if __name__ == '__main__':
    run_all_experiments()
