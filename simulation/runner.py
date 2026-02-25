import os
import sys
import requests
import time
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.environment import PhysicalEnvironment, DisturbanceGenerator
from analytics.evaluator import EvaluationAgent
from analytics.persistence import PersistenceAgent
import numpy as np

def run_simulation(steps=200, dt=0.1):
    env = PhysicalEnvironment(dt=dt)
    dist_gen = DisturbanceGenerator(probability=0.1, magnitude=2.0)
    eval_agent = EvaluationAgent()
    persistence = PersistenceAgent()
    
    url_update = "http://127.0.0.1:5000/update"
    url_train = "http://127.0.0.1:5000/train"
    
    y_ref_profile = [10.0 if i < 100 else 20.0 for i in range(steps)]
    
    results = {
        "time": [],
        "y_ref": [],
        "y_adaptive": [],
        "y_pid": [],
        "u_adaptive": [],
        "u_pid": []
    }
    
    # State for the adaptive simulation
    y_adaptive = 0.0
    # State for the PID simulation (separate env for fair comparison)
    env_pid = PhysicalEnvironment(dt=dt)
    y_pid = 0.0
    
    print("Starting Simulation...")
    
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    for t in range(steps):
        y_ref = y_ref_profile[t]
        disturbance = dist_gen.get_disturbance()
        
        # 1. Get Control Signals from Backend
        try:
            resp = requests.post(url_update, json={
                "y_ref": float(y_ref), 
                "y_actual": float(y_adaptive), 
                "y_pid": float(y_pid),
                "dt": dt
            })
            control_signals = resp.json()
        except:
            print("Backend not running? Start backend/app.py first.")
            return

        u_adaptive = control_signals["u_adaptive"]
        u_pid = control_signals["u_pid"]
        
        # 2. Apply Adaptive Control
        y_adaptive_new = env.step(u_adaptive, disturbance)
        dy = y_adaptive_new - y_adaptive
        
        # 3. Train Adaptive Agent (Online)
        requests.post(url_train, json={
            "y_ref": y_ref, 
            "y_actual_new": y_adaptive_new,
            "du": u_adaptive,
            "dy": dy,
            "dt": dt
        })
        
        # 4. Apply PID Control (Baseline)
        y_pid = env_pid.step(u_pid, disturbance)
        
        # Update states
        y_adaptive = y_adaptive_new
        
        # Save results
        results["time"].append(t * dt)
        results["y_ref"].append(y_ref)
        results["y_adaptive"].append(y_adaptive)
        results["y_pid"].append(y_pid)
        results["u_adaptive"].append(u_adaptive)
        results["u_pid"].append(u_pid)
        
        # Real-time Plotting (Every 5 steps for performance)
        if t % 5 == 0:
            ax[0].cla()
            ax[0].plot(results["time"], results["y_ref"], 'r--', label="Reference")
            ax[0].plot(results["time"], results["y_adaptive"], 'b-', label="Adaptive Control")
            ax[0].plot(results["time"], results["y_pid"], 'g-', label="PID Control")
            ax[0].set_title("System Output Comparison")
            ax[0].legend()
            
            ax[1].cla()
            ax[1].plot(results["time"], results["u_adaptive"], 'b-', label="U (Adaptive)")
            ax[1].plot(results["time"], results["u_pid"], 'g-', label="U (PID)")
            ax[1].set_title("Control Signals")
            ax[1].legend()
            
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    
    # Generate and Save Report
    metrics = eval_agent.generate_report(results, dt)
    persistence.save_run(metrics, results)
    
    return results

if __name__ == "__main__":
    # Note: Requires backend/app.py to be running in another process
    run_simulation()
