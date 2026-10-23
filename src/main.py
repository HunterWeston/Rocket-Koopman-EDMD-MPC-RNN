import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulator import run_monte_carlo, make_dirs
from parameters import cfg
from visualization import generate_3d_visualization_html


def compute_errors(hist):
    h = np.array(hist['h']);      h_ref = np.array(hist['h_ref'])
    yp = np.array(hist['y'])
    xp = np.array(hist.get('xp', np.zeros_like(yp)))
    vy = np.array(hist['vy'])
    vx = np.array(hist.get('vx', np.zeros_like(vy)))
    lateral_dist = np.sqrt(xp**2 + yp**2)
    lateral_vel  = np.sqrt(vx**2 + vy**2)
    theta = np.array(hist['theta'])
    pitch = np.array(hist.get('pitch', np.zeros_like(theta)))
    tilt_total = np.sqrt(theta**2 + pitch**2)
    vz = np.array(hist['vz']);    vz_ref = np.array(hist['vz_ref'])
    m  = np.array(hist['m'])
    m_initial = m[0]; m_final = m[-1]
    return {
        "Altitude MAE [m]":           float(np.mean(np.abs(h - h_ref))),
        "Altitude RMS [m]":           float(np.sqrt(np.mean((h - h_ref)**2))),
        "Lateral Drift Y RMS [m]":    float(np.sqrt(np.mean(yp**2))),
        "Max Lateral Drift Y [m]":    float(np.max(np.abs(yp))),
        "Lateral Drift 3D RMS [m]":   float(np.sqrt(np.mean(lateral_dist**2))),
        "Max Lateral Drift 3D [m]":   float(np.max(lateral_dist)),
        "Lateral Velocity RMS [m/s]": float(np.sqrt(np.mean(lateral_vel**2))),
        "Max Lateral Velocity [m/s]": float(np.max(lateral_vel)),
        "Tilt RMS [deg]":             float(np.degrees(np.sqrt(np.mean(tilt_total**2)))),
        "Max Tilt [deg]":             float(np.degrees(np.max(tilt_total))),
        "Vertical Velocity MAE [m/s]":float(np.mean(np.abs(vz - vz_ref))),
        "Apogee [m]":                 float(np.max(h)),
        "Flight Time [s]":            float(hist['t'][-1]),
        "Initial Mass [kg]":          float(m_initial),
        "Final Mass [kg]":            float(m_final),
        "Mass Burned [kg]":           float(m_initial - m_final),
    }
def plot_comparison_charts(pid_hists, mpc_hists, cfg, outdir, waypoint=None):
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    pid_hists = pid_hists or []
    mpc_hists = mpc_hists or []
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    has_pid = bool(pid_hists)
    title = "Koopman MPC vs PID — 3D Rocket Ascent (Monte Carlo)" if has_pid else "Koopman MPC — 3D Rocket Ascent (Waypoint Run)"
    fig.suptitle(title, fontsize=15, fontweight='bold')
    from matplotlib.lines import Line2D

    def _mc(ax, hists, key, label, color, alpha_ind=0.2, lw=2.2):
        for h in hists:
            ax.plot(h['t'], h[key], color=color, alpha=alpha_ind, lw=0.8)
        ml = min(len(h[key]) for h in hists)
        ax.plot(hists[0]['t'][:ml],
                np.array([h[key][:ml] for h in hists]).mean(0),
                color=color, lw=lw, label=label)

    if has_pid:
        _mc(axes[0,0], pid_hists, 'h', 'PID', '#e74c3c')
    _mc(axes[0,0], mpc_hists, 'h', 'MPC', '#2980b9')
    ref_hists = pid_hists if has_pid else mpc_hists
    ml = min(len(h['h_ref']) for h in ref_hists)
    axes[0,0].plot(ref_hists[0]['t'][:ml], ref_hists[0]['h_ref'][:ml],
                   'k--', lw=1.8, label='Reference', zorder=10)
    axes[0,0].set_title("Altitude Tracking"); axes[0,0].set_ylabel("Altitude [m]")
    axes[0,0].set_xlabel("Time [s]"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.35)

    ax = axes[0,1]
    if has_pid:
        for h in pid_hists:
            xp = np.array(h.get('xp', np.zeros(len(h['y']))))
            ax.plot(h['t'], xp, color='#e74c3c', alpha=0.15, lw=0.7)
            ax.plot(h['t'], np.array(h['y']), color='#e07020', alpha=0.15, lw=0.7)
    for h in mpc_hists:
        xp = np.array(h.get('xp', np.zeros(len(h['y']))))
        ax.plot(h['t'], xp, color='#2980b9', alpha=0.15, lw=0.7)
        ax.plot(h['t'], np.array(h['y']), color='#27ae60', alpha=0.15, lw=0.7)
    ml_m = min(len(h['y']) for h in mpc_hists)
    t_m = mpc_hists[0]['t'][:ml_m]
    xp_mpc = np.array([np.array(h.get('xp', np.zeros(ml_m)))[:ml_m] for h in mpc_hists]).mean(0)
    y_mpc  = np.array([h['y'][:ml_m] for h in mpc_hists]).mean(0)
    lat_mpc = np.sqrt(xp_mpc**2 + y_mpc**2)
    ax.plot(t_m, lat_mpc, color='#2980b9', lw=2.0, label='MPC total lateral')
    ax.axhline(0, color='lime', ls=':', lw=1.0, label='0 m')
    if waypoint is not None:
        def _wp_lat(wp, t_series=None):
            if isinstance(wp, (list, tuple)):
                if len(wp) >= 3:
                    return np.sqrt(float(wp[1])**2 + float(wp[2])**2)
                if len(wp) == 2:
                    return float(abs(wp[1]))
                return 0.0
            if isinstance(wp, dict):
                kind = wp.get('kind', 'stationary')
                if kind == 'moving' and t_series is not None:
                    n0 = float(wp.get('N0', 0.0)); e0 = float(wp.get('E0', 0.0))
                    vN = float(wp.get('vN', 0.0)); vE = float(wp.get('vE', 0.0))
                    return np.sqrt((n0 + vN * t_series)**2 + (e0 + vE * t_series)**2)
                if kind == 'stationary':
                    n = float(wp.get('N', 0.0)); e = float(wp.get('E', 0.0))
                    return float(np.sqrt(n*n + e*e))
                return 0.0
            return 0.0
        t_wp = t_m if not has_pid else pid_hists[0]['t'][:ml_m]
        wp_lat_p = _wp_lat(waypoint, t_wp)
        if np.ndim(wp_lat_p) == 0:
            ax.axhline(float(wp_lat_p), color='gold', ls=':', lw=1.5,
                       label=f'WP lateral={float(wp_lat_p):.0f}m')
        else:
            ax.plot(t_wp, wp_lat_p, color='gold', ls=':', lw=1.5,
                    label='WP lateral')
    ax.set_title("Total Lateral Displacement")
    ax.set_ylabel("Displacement [m]")
    ax.set_xlabel("Time [s]"); ax.legend(fontsize=7, ncol=2 if has_pid else 1); ax.grid(True, alpha=0.35)

    def _tilt(h):
        yaw = np.array(h['theta']); pit = np.array(h.get('pitch', np.zeros_like(yaw)))
        return np.degrees(np.sqrt(pit**2 + yaw**2))
    if has_pid:
        for h in pid_hists: axes[1,0].plot(h['t'], _tilt(h), color='#e74c3c', alpha=0.2, lw=0.8)
    for h in mpc_hists: axes[1,0].plot(h['t'], _tilt(h), color='#2980b9', alpha=0.2, lw=0.8)
    ml = min(len(h['theta']) for h in mpc_hists)
    if has_pid:
        ml = min(ml, min(len(h['theta']) for h in pid_hists))
        axes[1,0].plot(pid_hists[0]['t'][:ml],
                       np.array([_tilt(h)[:ml] for h in pid_hists]).mean(0),
                       color='#e74c3c', lw=2.2, label='PID mean |tilt|')
    axes[1,0].plot(mpc_hists[0]['t'][:ml],
                   np.array([_tilt(h)[:ml] for h in mpc_hists]).mean(0),
                   color='#2980b9', lw=2.2, label='MPC mean |tilt|')
    axes[1,0].axhline(0, color='k', ls='--', lw=1.2)
    axes[1,0].set_title("3D Tilt √(pitch²+yaw²)"); axes[1,0].set_ylabel("Tilt [deg]")
    axes[1,0].set_xlabel("Time [s]"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.35)

    if has_pid:
        for h in pid_hists: axes[1,1].step(h['t'], h['F'], color='#e74c3c', alpha=0.2, lw=0.7, where='post')
    for h in mpc_hists: axes[1,1].step(h['t'], h['F'], color='#2980b9', alpha=0.2, lw=0.7, where='post')
    axes[1,1].axhline(cfg.phys.m_dry * cfg.phys.g0, color='lime', ls=':', lw=1.5, label='Hover')
    legend_handles = [Line2D([0],[0],color='#2980b9',label='MPC'), Line2D([0],[0],color='lime',ls=':',label='Hover')]
    if has_pid:
        legend_handles.insert(0, Line2D([0],[0],color='#e74c3c',label='PID'))
    axes[1,1].legend(handles=legend_handles)
    axes[1,1].set_title("Thrust F(t)"); axes[1,1].set_ylabel("Thrust [N]")
    axes[1,1].set_xlabel("Time [s]"); axes[1,1].grid(True, alpha=0.35)

    if has_pid:
        for h in pid_hists:
            axes[2,0].plot(h['t'], np.degrees(np.array(h['delta'])), color='#e74c3c', alpha=0.18, lw=0.8)
            dp = np.degrees(np.array(h.get('delta_p', np.zeros(len(h['t'])))))
            axes[2,0].plot(h['t'], dp, color='#e74c3c', alpha=0.10, lw=0.6, ls='--')
    for h in mpc_hists:
        axes[2,0].plot(h['t'], np.degrees(np.array(h['delta'])), color='#2980b9', alpha=0.18, lw=0.8)
        dp = np.degrees(np.array(h.get('delta_p', np.zeros(len(h['t'])))))
        axes[2,0].plot(h['t'], dp, color='#2980b9', alpha=0.10, lw=0.6, ls='--')
    axes[2,0].axhline(0, color='k', ls='--', lw=1.0)
    axes[2,0].set_title("Gimbal: δ_yaw/East (solid) + δ_pitch/North (dashed)")
    axes[2,0].set_ylabel("Gimbal [deg]"); axes[2,0].set_xlabel("Time [s]")
    gimbal_handles = [Line2D([0],[0],color='#2980b9',label='MPC yaw'),
                      Line2D([0],[0],color='#999',ls='--',label='pitch channel')]
    if has_pid:
        gimbal_handles.insert(0, Line2D([0],[0],color='#e74c3c',label='PID yaw'))
    axes[2,0].legend(handles=gimbal_handles)
    axes[2,0].grid(True, alpha=0.35)

    ax6 = axes[2,1]
    if has_pid:
        for h in pid_hists:
            ax6.plot(h['y'], h.get('xp', [0]*len(h['y'])), color='#e74c3c', alpha=0.2, lw=0.8)
    for h in mpc_hists:
        ax6.plot(h['y'], h.get('xp', [0]*len(h['y'])), color='#2980b9', alpha=0.2, lw=0.8)
    mm = min(len(h['h']) for h in mpc_hists)
    if has_pid:
        mp = min(len(h['h']) for h in pid_hists)
        ax6.plot(np.array([h['y'][:mp] for h in pid_hists]).mean(0),
                 np.array([np.array(h.get('xp', np.zeros(mp)))[:mp] for h in pid_hists]).mean(0),
                 color='#e74c3c', lw=2.5, label='PID mean')
    ax6.plot(np.array([h['y'][:mm] for h in mpc_hists]).mean(0),
             np.array([np.array(h.get('xp', np.zeros(mm)))[:mm] for h in mpc_hists]).mean(0),
             color='#2980b9', lw=2.5, label='MPC mean')
    ax6.plot(0, 0, 'g*', ms=14, zorder=10, label='Launch')
    if waypoint is not None:
        wp_e, wp_n = float(waypoint[2]), float(waypoint[1])
        ax6.plot(wp_e, wp_n, 'y^', ms=14, zorder=11,
                 label=f'WP h={float(waypoint[0]):.0f}m N={wp_n:.0f} E={wp_e:.0f}')
        ax6.add_patch(plt.Circle((wp_e, wp_n), 10.0, color='yellow', fill=False,
                                  ls='--', lw=1.2, alpha=0.6))
    all_wvx = [float(np.mean(h.get('wind_vx', [0.0]))) for h in pid_hists + mpc_hists if 'wind_vx' in h]
    all_wvy = [float(np.mean(h.get('wind_vy', [0.0]))) for h in pid_hists + mpc_hists if 'wind_vy' in h]
    if all_wvx and all_wvy:
        mean_vx = float(np.mean(all_wvx))
        mean_vy = float(np.mean(all_wvy))
        mean_az = float(np.arctan2(mean_vy, mean_vx))
        all_vals = ax6.get_xlim() if ax6.get_xlim() != (0.0, 1.0) else (-20, 20)
        span = max(abs(ax6.get_xlim()[1] - ax6.get_xlim()[0]),
                   abs(ax6.get_ylim()[1] - ax6.get_ylim()[0]), 10.0)
        alen = span * 0.20
        ax6.annotate('', xy=(alen * np.sin(mean_az), alen * np.cos(mean_az)),
                     xytext=(0.0, 0.0),
                     arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))
        ax6.text(alen * np.sin(mean_az) * 1.25, alen * np.cos(mean_az) * 1.25,
                 f'Wind\n{np.degrees(mean_az):.0f}°', color='orange',
                 fontsize=8, ha='center', va='center', fontweight='bold')
    ax6.set_aspect('equal', adjustable='datalim')
    ax6.set_title("Top-Down Trajectory (East vs North)"); ax6.set_xlabel("East [m]")
    ax6.set_ylabel("North [m]"); ax6.legend(fontsize=7); ax6.grid(True, alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "plots", "pid_vs_mpc_comparison.png" if has_pid else "mpc_waypoint_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison chart: {p}")
def print_waypoint_transcript(mpc_hists, waypoint):
    """Print a table showing when each MPC run crossed closest to the 3D waypoint."""
    if waypoint is None:
        return
    h_pass, xp_tgt, y_tgt = float(waypoint[0]), float(waypoint[1]), float(waypoint[2])
    #threshold scales with target distance
    horiz_dist = float(np.sqrt(xp_tgt**2 + y_tgt**2))
    threshold = max(horiz_dist * 0.15, 3.0)
    print(f"\n  Waypoint crossing transcript")
    print(f"  Target: h={h_pass:.1f}m  N={xp_tgt:.1f}m  E={y_tgt:.1f}m  "
          f"(hit threshold = {threshold:.1f}m)")
    print()
    for run_i, hist in enumerate(mpc_hists):
        h_arr  = np.array(hist['h'])
        t_arr  = np.array(hist['t'])
        xp_arr = np.array(hist.get('xp', np.zeros_like(h_arr)))
        y_arr  = np.array(hist['y'])
        #3d distance to waypoint at every timestep
        dist = np.sqrt((xp_arr - xp_tgt)**2 + (y_arr - y_tgt)**2 + (h_arr - h_pass)**2)
        idx_close = int(np.argmin(dist))
        #timesteps within +/-40m of waypoint altitude
        near = np.where(np.abs(h_arr - h_pass) < 40.0)[0]
        hit_str = f"  ✓ WITHIN {threshold:.1f}m" if dist[idx_close] < threshold else ""
        print(f"  Run {run_i+1}  closest approach: t={t_arr[idx_close]:.2f}s  "
              f"h={h_arr[idx_close]:.1f}m  N={xp_arr[idx_close]:.1f}m  "
              f"E={y_arr[idx_close]:.1f}m  dist={dist[idx_close]:.1f}m{hit_str}")
        if len(near) > 0:
            print(f"  {'t[s]':>6}  {'h[m]':>7}  {'N[m]':>8}  {'E[m]':>8}  {'dist[m]':>8}")
            print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
            step = max(1, len(near) // 12)
            shown = set()
            for idx in near[::step]:
                shown.add(idx)
                marker = " ←" if idx == idx_close else ""
                print(f"  {t_arr[idx]:6.2f}  {h_arr[idx]:7.1f}  {xp_arr[idx]:8.1f}  "
                      f"{y_arr[idx]:8.1f}  {dist[idx]:8.1f}{marker}")
            if idx_close not in shown:
                print(f"  {t_arr[idx_close]:6.2f}  {h_arr[idx_close]:7.1f}  "
                      f"{xp_arr[idx_close]:8.1f}  {y_arr[idx_close]:8.1f}  "
                      f"{dist[idx_close]:8.1f}  ← closest")
        print()


#MAIN RUNNER
def run_and_compare(cfg, waypoint=None):
    """
    waypoint: 3D point the MPC tries to cross.
      None                → fly straight up (benchmark comparison vs PID)
      [h_pass, xN, yE]    → steer to (xN m North, yE m East) at altitude h_pass m
    PID always flies straight up.
    Both PID and MPC use IDENTICAL random seeds per run (same wind conditions).
    """
    if waypoint is not None:
        print(f"MPC waypoint target: h={waypoint[0]:.1f}m N={waypoint[1]:.1f}m E={waypoint[2]:.1f}m")
    import time as _time_
    shared_seed = int(cfg.sim.seed) if cfg.sim.seed is not None else (int(_time_.time() * 1000) % 999983)
    histories, all_metrics = {}, {}
    controllers = ["MPC"] if waypoint is not None else ["PID", "MPC"]
    for ctype in controllers:
        print(f"\nRunning {ctype} monte carlo ({cfg.sim.N_MC_runs} runs)...")
        kw = {"waypoint": waypoint} if ctype == "MPC" else {}
        all_hist, _ = run_monte_carlo(cfg, controller_type=ctype, shared_base_seed=shared_seed, **kw)
        avg = {}
        for hist in all_hist:
            for k, v in compute_errors(hist).items():
                avg[k] = avg.get(k, 0.0) + v / len(all_hist)
        all_metrics[ctype] = avg
        histories[ctype] = all_hist
        print(f"\n  {ctype} Performance Summary:")
        print(f"  {'Metric':<30} {'Value':>10}")
        print("  " + "=" * 42)
        for metric, value in avg.items():
            print(f"  {metric:<30} {value:>10.4f}")

    if "PID" in all_metrics and "MPC" in all_metrics:
        print("\n  === MPC vs PID ===")
        for k in ["Altitude MAE [m]", "Lateral Drift Y RMS [m]", "Tilt RMS [deg]", "Apogee [m]"]:
            p, m = all_metrics["PID"].get(k, 1), all_metrics["MPC"].get(k, 0)
            if p > 0:
                print(f"  {k:<30}: {100*(1-m/p):+.1f}%")

    make_dirs(cfg.out.outdir)
    plot_comparison_charts(histories.get("PID"), histories.get("MPC", []), cfg, cfg.out.outdir, waypoint=waypoint)
    out = generate_3d_visualization_html(histories, cfg, cfg.out.outdir, waypoint=waypoint)
    if out:
        print(f"\n3D visualization saved: {out}")
    if waypoint is not None:
        print_waypoint_transcript(histories.get("MPC", []), waypoint)


if __name__ == "__main__":
    print("=" * 58)
    print("Koopman MPC rocket sim")
    print("3D dynamics + wind + TVC + EDMD-c")
    print("=" * 58)
    #WAYPOINT OPTIONS
    #WAYPOINT = None
    #WAYPOINT = [200.0, -15.0, 40.0]
    WAYPOINT = [200.0, -15.0, 40.0]  #set None or [h_m, north_m, east_m]
    run_and_compare(cfg, waypoint=WAYPOINT)
    print("\ndone")
