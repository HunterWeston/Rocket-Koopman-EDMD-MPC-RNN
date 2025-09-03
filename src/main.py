#Hunter Weston

#Main

import matplotlib.pyplot as plt
from simulator import run_monte_carlo
from parameters import cfg
from rocket_model import rocket_altitude_parabola
import numpy as np

def plot_mc_results(all_hist, cfg):
    s_ref, v_ref, _ = rocket_altitude_parabola(cfg.sim.T_eval, apogee=80.0)

    plt.figure(figsize=(12, 6))
    for hist in all_hist:
        plt.plot(hist['t'], hist['h'], color='blue', alpha=0.5)
    t_vals = np.linspace(0, cfg.sim.T_eval, 500)
    plt.plot(t_vals, [s_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference Altitude')
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Monte Carlo Rocket Altitude Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for hist in all_hist:
        plt.plot(hist['t'], hist['v'], color='green', alpha=0.5)
    plt.plot(t_vals, [v_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference Velocity')
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Monte Carlo Rocket Velocity Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_errors(hist, cfg):
    s_ref, v_ref, _ = rocket_altitude_parabola(cfg.sim.T_eval, apogee=80.0) #reference trajectroy
    h_errors = [abs(h - s_ref(t)) for h, t in zip(hist['h'], hist['t'])]
    v_errors = [abs(v - v_ref(t)) for v, t in zip(hist['v'], hist['t'])]
    return {
        "Altitude MAE": sum(h_errors) / len(h_errors),
        "Velocity MAE": sum(v_errors) / len(v_errors),
        "Final Altitude Error": abs(hist['h'][-1] - s_ref(hist['t'][-1])),
        "Final Velocity Error": abs(hist['v'][-1] - v_ref(hist['t'][-1])),
    }

def plot_mc_results_side_by_side(pid_hist, mpc_hist, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Monte Carlo Results: PID vs MPC", fontsize=16)

    #alt vs time
    for hist in pid_hist:
        axes[0, 0].plot(hist['t'], hist['h'], color='blue', alpha=0.5)
    t_vals = np.linspace(0, cfg.sim.T_eval, 500)
    s_ref, _, _ = rocket_altitude_parabola(cfg.sim.T_eval, apogee=80.0)
    axes[0, 0].plot(t_vals, [s_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference')
    axes[0, 0].set_title("PID: Altitude vs Time")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Altitude [m]")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for hist in mpc_hist:
        axes[0, 1].plot(hist['t'], hist['h'], color='blue', alpha=0.5)
    axes[0, 1].plot(t_vals, [s_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference')
    axes[0, 1].set_title("MPC: Altitude vs Time")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Altitude [m]")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    #velo vs time
    for hist in pid_hist:
        axes[1, 0].plot(hist['t'], hist['v'], color='green', alpha=0.5)
    _, v_ref, _ = rocket_altitude_parabola(cfg.sim.T_eval, apogee=80.0)
    axes[1, 0].plot(t_vals, [v_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference')
    axes[1, 0].set_title("PID: Velocity vs Time")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Velocity [m/s]")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for hist in mpc_hist:
        axes[1, 1].plot(hist['t'], hist['v'], color='green', alpha=0.5)
    axes[1, 1].plot(t_vals, [v_ref(t) for t in t_vals], color='red', linestyle='--', label='Reference')
    axes[1, 1].set_title("MPC: Velocity vs Time")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Velocity [m/s]")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def run_and_compare(cfg):
    results = {}
    histories = {}

    for controller_type in ["PID", "MPC"]:
        print(f"Running Monte Carlo simulations with {controller_type} controller...")
        all_hist, _ = run_monte_carlo(cfg, controller_type=controller_type)

        #avg error
        avg_errors = {"Altitude MAE": 0, "Velocity MAE": 0, "Final Altitude Error": 0, "Final Velocity Error": 0}
        for hist in all_hist:
            errors = compute_errors(hist, cfg)
            for key in avg_errors:
                avg_errors[key] += errors[key]
        for key in avg_errors:
            avg_errors[key] /= len(all_hist)

        results[controller_type] = avg_errors
        histories[controller_type] = all_hist

        #display metrics
        print(f"\n{controller_type} Controller Metrics:")
        print(f"{'Metric':<25}{'Value':<15}")
        print("=" * 40)
        for metric, value in avg_errors.items():
            print(f"{metric:<25}{value:<15.4f}")

    plot_mc_results_side_by_side(histories["PID"], histories["MPC"], cfg)

if __name__ == "__main__":
    print("Running simulations...")
    run_and_compare(cfg)