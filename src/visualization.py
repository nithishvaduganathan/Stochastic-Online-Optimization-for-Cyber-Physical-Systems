"""
Visualization Module

This module provides visualization functions for analyzing the performance
of the online learning controller, including:
- Temperature tracking plots
- Convergence analysis
- Parameter evolution
- Comparative analysis between controllers
- Statistical performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os


def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


def plot_temperature_tracking(
    time: np.ndarray,
    reference: np.ndarray,
    actual: np.ndarray,
    title: str = "Temperature Tracking Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temperature tracking performance.
    
    Args:
        time: Time array
        reference: Reference temperature signal
        actual: Actual temperature response
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Temperature tracking
    axes[0].plot(time, reference, 'b--', label='Reference', linewidth=2)
    axes[0].plot(time, actual, 'r-', label='Actual', linewidth=1.5)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Tracking error
    error = reference - actual
    axes[1].plot(time, error, 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Tracking Error (°C)')
    axes[1].set_title('Tracking Error Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_convergence_analysis(
    controller_history: Dict[str, np.ndarray],
    title: str = "Online Learning Convergence Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot convergence analysis of the online learning controller.
    
    Args:
        controller_history: Dictionary with 'parameters', 'errors', 'learning_rates'
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Parameter evolution
    params = controller_history['parameters']
    axes[0, 0].plot(params[:, 0], 'b-', label='Gain Correction', linewidth=1.5)
    axes[0, 0].plot(params[:, 1], 'r-', label='Bias Correction', linewidth=1.5)
    if params.shape[1] > 2:
        axes[0, 0].plot(params[:, 2], 'g-', label='Integral Gain', linewidth=1.5)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].set_title('Parameter Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error over time (with moving average)
    errors = controller_history['errors']
    window = min(50, len(errors) // 10 + 1)
    if len(errors) > window:
        moving_avg = np.convolve(np.abs(errors), np.ones(window)/window, mode='valid')
        time_ma = np.arange(window-1, len(errors))
        axes[0, 1].plot(np.abs(errors), 'b-', alpha=0.3, linewidth=0.5, label='Instantaneous |Error|')
        axes[0, 1].plot(time_ma, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
    else:
        axes[0, 1].plot(np.abs(errors), 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('|Error| (°C)')
    axes[0, 1].set_title('Absolute Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate decay
    lr = controller_history['learning_rates']
    axes[1, 0].plot(lr, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative squared error
    cumulative_se = np.cumsum(errors ** 2)
    axes[1, 1].plot(cumulative_se, 'purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cumulative Squared Error')
    axes[1, 1].set_title('Cumulative Squared Error (Convergence Indicator)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_controller_comparison(
    time: np.ndarray,
    reference: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "Controller Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare performance of different controllers.
    
    Args:
        time: Time array
        reference: Reference signal
        results: Dictionary mapping controller names to their results
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    n_controllers = len(results)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_controllers + 1))
    
    # Plot reference
    axes[0].plot(time, reference, 'k--', label='Reference', linewidth=2.5, alpha=0.8)
    
    # Plot each controller's response
    for idx, (name, data) in enumerate(results.items()):
        temp = data['temperature']
        # Ensure temp has same length as time
        plot_len = min(len(time), len(temp))
        axes[0].plot(time[:plot_len], temp[:plot_len], 
                    color=colors[idx], label=name, linewidth=1.5)
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Temperature Response Comparison')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot error comparison
    for idx, (name, data) in enumerate(results.items()):
        errors = data['errors']
        plot_len = min(len(time), len(errors))
        axes[1].plot(time[:plot_len], np.abs(errors[:plot_len]), 
                    color=colors[idx], label=name, linewidth=1.5, alpha=0.7)
    
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('|Error| (°C)')
    axes[1].set_title('Absolute Error Comparison')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_disturbance_response(
    time: np.ndarray,
    temperature: np.ndarray,
    reference: np.ndarray,
    disturbances: np.ndarray,
    ambient_changes: Optional[List[Tuple[int, float]]] = None,
    title: str = "Disturbance Response Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot system response to disturbances and environmental changes.
    
    Args:
        time: Time array
        temperature: Temperature response
        reference: Reference signal
        disturbances: Disturbance history
        ambient_changes: List of (time, new_ambient) tuples
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Temperature tracking
    axes[0].plot(time, reference, 'b--', label='Reference', linewidth=2)
    axes[0].plot(time, temperature, 'r-', label='Actual', linewidth=1.5)
    
    # Mark ambient changes
    if ambient_changes:
        for t, new_ambient in ambient_changes:
            axes[0].axvline(x=t, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            axes[0].annotate(f'Ambient→{new_ambient}°C', 
                           xy=(t, axes[0].get_ylim()[1]),
                           xytext=(t+5, axes[0].get_ylim()[1]),
                           fontsize=10)
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Temperature Response')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Disturbances
    axes[1].plot(time[:len(disturbances)], disturbances, 'g-', linewidth=1)
    axes[1].fill_between(time[:len(disturbances)], disturbances, alpha=0.3)
    axes[1].set_ylabel('Disturbance')
    axes[1].set_title('Stochastic Disturbances')
    axes[1].grid(True, alpha=0.3)
    
    # Error
    plot_len = min(len(time), len(temperature), len(reference))
    error = reference[:plot_len] - temperature[:plot_len]
    axes[2].plot(time[:plot_len], error, 'purple', linewidth=1.5)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Tracking Error (°C)')
    axes[2].set_title('Tracking Error')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_performance_metrics(
    results: Dict[str, Dict[str, np.ndarray]],
    metrics: List[str] = ['rmse', 'mae', 'max_error', 'settling_time'],
    title: str = "Performance Metrics Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar charts comparing performance metrics.
    
    Args:
        results: Dictionary mapping controller names to their results
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    # Compute metrics
    computed_metrics = {}
    for name, data in results.items():
        errors = data['errors']
        computed_metrics[name] = {
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'settling_time': _compute_settling_time(errors, threshold=0.5)
        }
    
    n_metrics = len(metrics)
    n_controllers = len(results)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_controllers)
    width = 0.6
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_controllers))
    
    for idx, metric in enumerate(metrics):
        values = [computed_metrics[name][metric] for name in results.keys()]
        bars = axes[idx].bar(x, values, width, color=colors)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(results.keys(), rotation=45, ha='right')
        axes[idx].set_title(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].annotate(f'{val:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def _compute_settling_time(errors: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute settling time (time to reach and stay within threshold).
    
    Args:
        errors: Error array
        threshold: Error threshold for settling
        
    Returns:
        Settling time (or -1 if never settles)
    """
    for i in range(len(errors) - 1, -1, -1):
        if np.abs(errors[i]) > threshold:
            return i + 1
    return 0


def print_performance_summary(results: Dict[str, Dict[str, np.ndarray]]):
    """
    Print a summary table of performance metrics.
    
    Args:
        results: Dictionary mapping controller names to their results
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Controller':<20} {'RMSE':>10} {'MAE':>10} {'Max Error':>12} {'Settling':>10}")
    print("-" * 70)
    
    for name, data in results.items():
        errors = data['errors']
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        max_err = np.max(np.abs(errors))
        settling = _compute_settling_time(errors, threshold=0.5)
        
        print(f"{name:<20} {rmse:>10.4f} {mae:>10.4f} {max_err:>12.4f} {settling:>10}")
    
    print("=" * 70)


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: str = "results"
):
    """
    Save all figures to the output directory.
    
    Args:
        figures: Dictionary mapping figure names to Figure objects
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Saved: {path}")
