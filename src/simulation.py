"""
Main Simulation Runner

This module provides the main simulation framework for evaluating
the online learning controller on the smart temperature control system.

Features:
- Multiple experimental scenarios
- Controller comparison studies
- Robustness evaluation under stochastic disturbances
- Adaptability testing with environmental changes
- Comprehensive result visualization

Usage:
    python -m src.simulation
    
Or import and use programmatically:
    from src.simulation import run_experiment, run_comparison_study
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys

from .thermal_system import ThermalSystem, ApproximateThermalModel
from .online_controller import OnlineLearningController, StaticController, PIDController
from .disturbances import (
    GaussianNoise, StepDisturbance, CombinedDisturbance,
    EnvironmentChange, create_realistic_disturbance
)
from .gradient_estimator import FiniteDifferenceGradient, SPSAGradient
from .visualization import (
    plot_temperature_tracking,
    plot_convergence_analysis,
    plot_controller_comparison,
    plot_disturbance_response,
    plot_performance_metrics,
    print_performance_summary,
    save_all_figures
)


def generate_reference_signal(
    n_steps: int,
    signal_type: str = 'step',
    base_temperature: float = 22.0,
    step_changes: Optional[List[Tuple[int, float]]] = None
) -> np.ndarray:
    """
    Generate reference temperature signal.
    
    Args:
        n_steps: Number of time steps
        signal_type: 'constant', 'step', 'ramp', or 'sinusoidal'
        base_temperature: Base temperature value
        step_changes: List of (time, new_value) for step signals
        
    Returns:
        Reference signal array
    """
    reference = np.ones(n_steps) * base_temperature
    
    if signal_type == 'constant':
        pass
    
    elif signal_type == 'step':
        if step_changes is None:
            step_changes = [(200, 25.0), (500, 20.0), (800, 24.0)]
        
        for step_time, new_value in step_changes:
            if step_time < n_steps:
                reference[step_time:] = new_value
    
    elif signal_type == 'ramp':
        # Gradual increase then decrease
        mid_point = n_steps // 2
        reference[:mid_point] = np.linspace(base_temperature, base_temperature + 5, mid_point)
        reference[mid_point:] = np.linspace(base_temperature + 5, base_temperature, n_steps - mid_point)
    
    elif signal_type == 'sinusoidal':
        # Sinusoidal reference (e.g., day-night temperature cycle)
        amplitude = 3.0
        period = n_steps / 2
        reference = base_temperature + amplitude * np.sin(2 * np.pi * np.arange(n_steps) / period)
    
    return reference


def run_experiment(
    n_steps: int = 1000,
    controller_type: str = 'online',
    disturbance_config: Optional[Dict] = None,
    environment_changes: Optional[List[EnvironmentChange]] = None,
    reference_signal: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run a single experiment with the specified configuration.
    
    Args:
        n_steps: Number of simulation steps
        controller_type: 'online', 'static', or 'pid'
        disturbance_config: Configuration for disturbance generator
        environment_changes: List of environment change events
        reference_signal: Custom reference signal (generated if None)
        seed: Random seed
        verbose: Print progress updates
        
    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)
    
    # System parameters (true system - unknown to controller)
    true_alpha = 0.1
    true_beta = 0.5
    initial_temp = 20.0
    ambient_temp = 15.0
    
    # Initialize true system
    system = ThermalSystem(
        alpha=true_alpha,
        beta=true_beta,
        initial_temperature=initial_temp,
        ambient_temperature=ambient_temp
    )
    
    # Initialize approximate model (used by controller - has errors)
    approx_model = ApproximateThermalModel(
        estimated_alpha=0.12,  # 20% error
        estimated_beta=0.45,   # 10% error
        ambient_temperature=ambient_temp
    )
    
    # Initialize controller
    if controller_type == 'online':
        controller = OnlineLearningController(
            approximate_model=approx_model,
            learning_rate=0.05,
            momentum=0.9,
            control_bounds=(-10.0, 10.0)
        )
    elif controller_type == 'static':
        controller = StaticController(
            approximate_model=approx_model,
            control_bounds=(-10.0, 10.0)
        )
    elif controller_type == 'pid':
        controller = PIDController(
            kp=1.5,
            ki=0.1,
            kd=0.1,
            control_bounds=(-10.0, 10.0)
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Create disturbance generator
    if disturbance_config is None:
        disturbance = create_realistic_disturbance(noise_std=0.2, seed=seed)
    else:
        disturbance = CombinedDisturbance(
            gaussian_noise=GaussianNoise(
                std=disturbance_config.get('noise_std', 0.2),
                seed=seed
            ),
            step_disturbances=disturbance_config.get('step_disturbances', []),
            periodic_disturbances=disturbance_config.get('periodic_disturbances', [])
        )
    
    # Generate reference signal
    if reference_signal is None:
        reference_signal = generate_reference_signal(n_steps, signal_type='step')
    
    # Environment change tracking
    ambient_change_times = []
    if environment_changes:
        for ec in environment_changes:
            ambient_change_times.append((ec.change_time, ec.new_ambient_temperature))
    
    # Simulation loop
    temperature_history = [system.get_state()]
    control_history = []
    disturbance_history = []
    
    for k in range(n_steps):
        # Check for environment changes
        if environment_changes:
            for ec in environment_changes:
                if ec.should_change(k):
                    system.set_ambient_temperature(ec.new_ambient_temperature)
                    if verbose:
                        print(f"Time {k}: Environment changed to {ec.new_ambient_temperature}°C")
        
        # Get current state and reference
        current_temp = system.get_state()
        ref_temp = reference_signal[k]
        
        # Compute control action
        if controller_type == 'online':
            control = controller.compute_control(current_temp, ref_temp, update_parameters=True)
        else:
            control = controller.compute_control(current_temp, ref_temp)
        
        # Get disturbance
        dist = disturbance.get_disturbance(k)
        
        # Step the system
        new_temp = system.step(control, dist)
        
        # Record history
        temperature_history.append(new_temp)
        control_history.append(control)
        disturbance_history.append(dist)
        
        # Progress update
        if verbose and (k + 1) % 200 == 0:
            print(f"Step {k+1}/{n_steps}: Temp={new_temp:.2f}°C, Ref={ref_temp:.2f}°C, Error={ref_temp-new_temp:.2f}°C")
    
    # Compile results
    results = {
        'temperature': np.array(temperature_history[:-1]),
        'control': np.array(control_history),
        'disturbance': np.array(disturbance_history),
        'reference': reference_signal,
        'errors': controller.get_history()['errors'],
        'ambient_changes': ambient_change_times
    }
    
    if controller_type == 'online':
        results['controller_history'] = controller.get_history()
    
    return results


def run_comparison_study(
    n_steps: int = 1000,
    seed: int = 42,
    output_dir: str = 'results',
    save_figures: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run a comparison study between different controllers.
    
    Args:
        n_steps: Number of simulation steps
        seed: Random seed
        output_dir: Directory for saving results
        save_figures: Whether to save figures
        
    Returns:
        Dictionary with results for each controller
    """
    print("\n" + "=" * 70)
    print("CONTROLLER COMPARISON STUDY")
    print("=" * 70)
    
    # Common reference signal
    reference = generate_reference_signal(n_steps, signal_type='step')
    
    # Environment changes (test adaptability)
    env_changes = [
        EnvironmentChange(change_time=300, new_ambient_temperature=10.0),
        EnvironmentChange(change_time=700, new_ambient_temperature=20.0),
    ]
    
    results = {}
    
    # Run experiments for each controller type
    for controller_type in ['online', 'static', 'pid']:
        print(f"\n--- Running {controller_type.upper()} Controller ---")
        results[controller_type] = run_experiment(
            n_steps=n_steps,
            controller_type=controller_type,
            environment_changes=env_changes,
            reference_signal=reference,
            seed=seed,
            verbose=True
        )
    
    # Print performance summary
    print_performance_summary(results)
    
    # Generate visualizations
    time = np.arange(n_steps)
    figures = {}
    
    # Comparison plot
    figures['controller_comparison'] = plot_controller_comparison(
        time=time,
        reference=reference,
        results=results,
        title="Controller Performance Comparison\n(with Environment Changes at t=300 and t=700)"
    )
    
    # Online controller convergence
    if 'controller_history' in results['online']:
        figures['convergence_analysis'] = plot_convergence_analysis(
            controller_history=results['online']['controller_history'],
            title="Online Learning Controller - Convergence Analysis"
        )
    
    # Disturbance response
    figures['disturbance_response'] = plot_disturbance_response(
        time=time,
        temperature=results['online']['temperature'],
        reference=reference,
        disturbances=results['online']['disturbance'],
        ambient_changes=results['online']['ambient_changes'],
        title="Online Controller - Disturbance Response"
    )
    
    # Performance metrics
    figures['performance_metrics'] = plot_performance_metrics(
        results=results,
        title="Performance Metrics Comparison"
    )
    
    # Save figures
    if save_figures:
        os.makedirs(output_dir, exist_ok=True)
        save_all_figures(figures, output_dir)
    
    return results


def run_robustness_study(
    n_trials: int = 10,
    n_steps: int = 500,
    noise_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
    output_dir: str = 'results'
) -> Dict[str, np.ndarray]:
    """
    Study controller robustness under different noise levels.
    
    Args:
        n_trials: Number of trials per noise level
        n_steps: Steps per trial
        noise_levels: List of noise standard deviations to test
        output_dir: Directory for saving results
        
    Returns:
        Dictionary with robustness study results
    """
    print("\n" + "=" * 70)
    print("ROBUSTNESS STUDY")
    print("=" * 70)
    
    results = {
        'noise_levels': np.array(noise_levels),
        'online_rmse': [],
        'online_rmse_std': [],
        'static_rmse': [],
        'static_rmse_std': [],
        'pid_rmse': [],
        'pid_rmse_std': [],
    }
    
    reference = generate_reference_signal(n_steps, signal_type='step',
                                         step_changes=[(100, 25.0), (300, 20.0)])
    
    for noise_std in noise_levels:
        print(f"\nTesting noise level: {noise_std}")
        
        online_errors = []
        static_errors = []
        pid_errors = []
        
        disturbance_config = {'noise_std': noise_std}
        
        for trial in range(n_trials):
            # Online controller
            res = run_experiment(
                n_steps=n_steps,
                controller_type='online',
                disturbance_config=disturbance_config,
                reference_signal=reference,
                seed=trial * 100,
                verbose=False
            )
            online_errors.append(np.sqrt(np.mean(res['errors'] ** 2)))
            
            # Static controller
            res = run_experiment(
                n_steps=n_steps,
                controller_type='static',
                disturbance_config=disturbance_config,
                reference_signal=reference,
                seed=trial * 100,
                verbose=False
            )
            static_errors.append(np.sqrt(np.mean(res['errors'] ** 2)))
            
            # PID controller
            res = run_experiment(
                n_steps=n_steps,
                controller_type='pid',
                disturbance_config=disturbance_config,
                reference_signal=reference,
                seed=trial * 100,
                verbose=False
            )
            pid_errors.append(np.sqrt(np.mean(res['errors'] ** 2)))
        
        results['online_rmse'].append(np.mean(online_errors))
        results['online_rmse_std'].append(np.std(online_errors))
        results['static_rmse'].append(np.mean(static_errors))
        results['static_rmse_std'].append(np.std(static_errors))
        results['pid_rmse'].append(np.mean(pid_errors))
        results['pid_rmse_std'].append(np.std(pid_errors))
        
        print(f"  Online RMSE: {np.mean(online_errors):.4f} ± {np.std(online_errors):.4f}")
        print(f"  Static RMSE: {np.mean(static_errors):.4f} ± {np.std(static_errors):.4f}")
        print(f"  PID RMSE: {np.mean(pid_errors):.4f} ± {np.std(pid_errors):.4f}")
    
    # Convert to arrays
    for key in ['online_rmse', 'online_rmse_std', 'static_rmse', 'static_rmse_std', 
                'pid_rmse', 'pid_rmse_std']:
        results[key] = np.array(results[key])
    
    # Plot robustness comparison
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(noise_levels, results['online_rmse'], yerr=results['online_rmse_std'],
                label='Online Learning', marker='o', capsize=5, linewidth=2)
    ax.errorbar(noise_levels, results['static_rmse'], yerr=results['static_rmse_std'],
                label='Static', marker='s', capsize=5, linewidth=2)
    ax.errorbar(noise_levels, results['pid_rmse'], yerr=results['pid_rmse_std'],
                label='PID', marker='^', capsize=5, linewidth=2)
    
    ax.set_xlabel('Noise Standard Deviation')
    ax.set_ylabel('RMSE')
    ax.set_title('Controller Robustness to Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'robustness_study.png'), 
                bbox_inches='tight', dpi=150)
    print(f"\nSaved robustness study plot to {output_dir}/robustness_study.png")
    
    return results


def main():
    """Main entry point for running simulations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Stochastic Online Optimization Simulation'
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['comparison', 'robustness', 'single'],
        default='comparison',
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=1000,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save figures'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STOCHASTIC ONLINE OPTIMIZATION FOR CYBER-PHYSICAL SYSTEMS")
    print("Smart Temperature Control Simulation")
    print("=" * 70)
    
    if args.experiment == 'comparison':
        run_comparison_study(
            n_steps=args.steps,
            seed=args.seed,
            output_dir=args.output,
            save_figures=not args.no_save
        )
    
    elif args.experiment == 'robustness':
        run_robustness_study(
            n_steps=args.steps,
            output_dir=args.output
        )
    
    elif args.experiment == 'single':
        print("\nRunning single experiment with online controller...")
        results = run_experiment(
            n_steps=args.steps,
            controller_type='online',
            seed=args.seed,
            verbose=True
        )
        
        # Quick visualization
        time = np.arange(args.steps)
        fig = plot_temperature_tracking(
            time=time,
            reference=results['reference'],
            actual=results['temperature'],
            title="Single Experiment - Online Learning Controller"
        )
        
        if not args.no_save:
            os.makedirs(args.output, exist_ok=True)
            fig.savefig(os.path.join(args.output, 'single_experiment.png'),
                       bbox_inches='tight', dpi=150)
    
    print("\nSimulation completed successfully!")
    import matplotlib.pyplot as plt
    plt.close('all')


if __name__ == '__main__':
    main()
