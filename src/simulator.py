import copy
import os
import time as _time

import numpy as np

from parameters import cfg
from rocket_model import Rocket3D, rocket_ascent_reference
from lifting import LiftingMap, fit_C
from edmd import EDMDc
from residual_nn import TinyMLP, ReplayBuffer
from mpc import LinearMPC


MPC_TUNING = {
    "q_y_scale": 2.6,
    "q_vy_scale": 2.0,
    "q_theta_min": 12.0,
    "q_omega": 1.5,
    "r_delta_scale": 0.50,
    "r_delta_min": 1.2,
    "lat_horizon": 14,
    "activate_time": 0.55,
    "thrust_margin": 1.03,
    "passed_wp_altitude_margin": 18.0,
    "wp_x_bias_base": -1.0,
    "wp_x_bias_wind": -0.5,
    "wp_x_bias_min": -6.0,
    "wp_x_bias_max": 3.0,
    "wp_y_bias_base": 8.0,
    "wp_y_bias_wind": -0.7,
    "wp_y_cross_wind_vx": 0.55,
    "wp_y_bias_min": -6.0,
    "wp_y_bias_max": 12.0,
    "tilt_soft_deg": 5.0,
    "tilt_hard_deg": 12.0,
    "tilt_cut_deg": 10.0,
    "tilt_cut_scale": 0.18,
    "lead_exp": 0.42,
    "terminal_start": 0.16,
    "terminal_span": 0.82,
    "t_rem_min": 0.38,
    "t_rem_max": 6.0,
    "delta_cap_deg": 12.0,
    "a_lat_min": 2.25,
    "a_lat_scale": 1.65,
    "ff_base": 0.78,
    "ff_terminal": 0.34,
    "a_x_pos": 0.46,
    "a_x_vel": 1.20,
    "a_y_pos": 0.70,
    "a_y_vel": 1.35,
    "overshoot_terminal": 0.55,
    "overshoot_scale": 1.45,
    "overshoot_bias": 0.35,
    "dp_wind_ff_scale": 0.65,
    "dy_wind_ff_scale": 0.85,
    "dp_wind_ff_cap_deg": 3.0,
    "dy_wind_ff_cap_deg": 4.0,
    "east_scale_wind_gain": 0.07,
    "east_scale_min": 0.55,
    "benchmark_model_p": 0.55,
    "benchmark_model_y": 0.65,
    "benchmark_ff_p": 0.35,
    "benchmark_ff_y": 0.35,
    "benchmark_int_decay": 0.996,
    "benchmark_int_pos_gain": 0.0014,
    "benchmark_int_cap": 0.016,
    "waypoint_model_p": 0.30,
    "waypoint_model_y": 0.40,
    "waypoint_ff_p": 0.75,
    "waypoint_ff_y": 1.00,
    "waypoint_int_p": 0.0080,
    "waypoint_int_y": 0.0170,
    "waypoint_int_cap_p": 0.055,
    "waypoint_int_cap_y": 0.070,
    "gimbal_noise_mpc": 0.0010,
}


class WindModel:

    def __init__(self, rng, cfg):
        sim = cfg.sim
        wind_speed = sim.wind_speed_ms

        self._azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        self._steady_speed = float(wind_speed)
        self._turb_std = 0.20 * wind_speed
        self._tau = 1.0
        self._w = 0.0
        self._gust_speed = 0.0
        self._gust_steps = 0
        self._gust_prob = sim.wind_gust_prob
        self._gust_max = wind_speed * sim.wind_gust_factor
        self._az_drift = 0.0
        self._az_tau = 3.0
        self._az_std = np.deg2rad(2.5)
        self._rng = rng

    @property
    def azimuth(self):
        return float((self._azimuth + self._az_drift) % (2.0 * np.pi))

    @property
    def speed(self):
        return self._steady_speed

    def step(self, dt):
        alpha = np.exp(-dt / self._tau)
        noise = self._rng.normal(0.0, self._turb_std * np.sqrt(max(1.0 - alpha**2, 0.0)))
        self._w = alpha * self._w + noise

        alpha_az = np.exp(-dt / self._az_tau)
        az_noise = self._rng.normal(0.0, self._az_std * np.sqrt(max(1.0 - alpha_az**2, 0.0)))
        self._az_drift = float(
            np.clip(alpha_az * self._az_drift + az_noise, -np.deg2rad(10.0), np.deg2rad(10.0))
        )

        if self._gust_steps > 0:
            self._gust_steps -= 1
        elif self._rng.random() < self._gust_prob:
            self._gust_speed = self._rng.uniform(-self._gust_max * 0.3, self._gust_max)
            self._gust_steps = int(self._rng.integers(6, 25))
        else:
            self._gust_speed = 0.0

        total_speed = self._steady_speed + self._w + self._gust_speed
        az = self.azimuth
        wind_vx = total_speed * np.cos(az)
        wind_vy = total_speed * np.sin(az)
        return float(wind_vx), float(wind_vy)


def get_srb_thrust(t, F_max, profile):
    """Interpolate SRB thrust from burn profile at time t."""
    if t <= profile[0][0]:
        return float(profile[0][1] * F_max)
    if t >= profile[-1][0]:
        return 0.0
    for i in range(len(profile) - 1):
        t0, f0 = profile[i]
        t1, f1 = profile[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / max(t1 - t0, 1e-9)
            return float((f0 + alpha * (f1 - f0)) * F_max)
    return 0.0


def compute_attitude_delta_pid(x, cfg):
    """
    Simple attitude PD used as the PID baseline.
    x = [h, vz, xp, vx, yp, vy, pitch, prate, yaw, yrate, m]
    Returns (delta_p, delta_y).
    """
    delta_p = cfg.pid.Kp_att * (-x[6]) + cfg.pid.Kd_att * (-x[7])
    delta_y = cfg.pid.Kp_att * (-x[8]) + cfg.pid.Kd_att * (-x[9])
    delta_p = float(np.clip(delta_p, -cfg.phys.delta_max, cfg.phys.delta_max))
    delta_y = float(np.clip(delta_y, -cfg.phys.delta_max, cfg.phys.delta_max))
    return delta_p, delta_y


class PIDController:
    def __init__(self, cfg, dt):
        self.dt = dt
        self.F_min = cfg.ctrl.F_min
        self.F_max = cfg.ctrl.F_max
        self.Kp = cfg.pid.Kp_alt
        self.Ki = cfg.pid.Ki_alt
        self.Kd = cfg.pid.Kd_alt
        self.integral = 0.0

    def step(self, x, h_ref, vz_ref, g0):
        h, vz, m = x[0], x[1], x[10]
        err = h_ref - h
        self.integral = np.clip(self.integral + err * self.dt, -60.0, 60.0)
        F = m * g0 + self.Kp * err + self.Ki * self.integral + self.Kd * (vz_ref - vz)
        return float(np.clip(F, self.F_min, self.F_max))


def gen_training_data(plant, lift, rng, dt, T, cfg):
    """3-state altitude training data for Koopman diagnostics."""
    steps = int(T / dt)
    X = np.zeros((steps + 1, 3))
    U = np.zeros((steps, 1))
    h0 = rng.uniform(0.0, 40.0)
    vz0 = rng.uniform(0.0, 15.0)
    m0 = cfg.phys.m_dry + rng.uniform(0.2, cfg.phys.m_fuel0)
    x3 = np.array([h0, vz0, m0])
    X[0] = x3
    for k in range(steps):
        F = rng.uniform(cfg.ctrl.F_min * 0.4, cfg.ctrl.F_max)
        U[k, 0] = F
        h, vz, m = x3
        m = max(m, cfg.phys.m_dry)
        Dz = 0.5 * cfg.phys.rho * vz * abs(vz) * cfg.phys.Cd * cfg.phys.A_ref
        vz_next = vz + dt * ((F - Dz) / m - cfg.phys.g0 + rng.normal(0.0, 0.15))
        h_next = max(h + dt * vz, 0.0)
        mdot = -F / (cfg.phys.Isp * cfg.phys.g0)
        if m <= cfg.phys.m_dry:
            mdot = 0.0
        m_next = max(m + dt * mdot, cfg.phys.m_dry)
        x3 = np.array([h_next, vz_next, m_next])
        X[k + 1] = x3
    Z = lift.lift(X[:-1])
    Zp = lift.lift(X[1:])
    return X[:-1], U, X[1:], Z, Zp


def gen_lateral_training_data(plant, lift_lat, rng, dt, T, cfg, axis="yaw"):
    """
    4-state one-axis lateral Koopman training data: [pos, vel, angle, rate].
    axis="yaw" uses East/Yaw and delta_y. axis="pitch" uses North/Pitch and delta_p.
    """
    steps = int(T / dt)
    X_lat = np.zeros((steps + 1, 4))
    U = np.zeros((steps, 1))

    if axis == "yaw":
        x = np.array([
            rng.uniform(15.0, 45.0),
            rng.uniform(18.0, 40.0),
            0.0,
            0.0,
            rng.uniform(-4.0, 4.0),
            rng.uniform(-1.0, 1.0),
            0.0,
            0.0,
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.03, 0.03),
            rng.uniform(cfg.phys.m_dry + 0.4, cfg.phys.m_dry + cfg.phys.m_fuel0),
        ], dtype=float)
        X_lat[0] = np.array([x[4], x[5], x[8], x[9]])
    else:
        x = np.array([
            rng.uniform(15.0, 45.0),
            rng.uniform(18.0, 40.0),
            rng.uniform(-4.0, 4.0),
            rng.uniform(-1.0, 1.0),
            0.0,
            0.0,
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.03, 0.03),
            0.0,
            0.0,
            rng.uniform(cfg.phys.m_dry + 0.4, cfg.phys.m_dry + cfg.phys.m_fuel0),
        ], dtype=float)
        X_lat[0] = np.array([x[2], x[3], x[6], x[7]])

    for k in range(steps):
        t_sim = 1.0 + (k * dt) % 8.0
        F = max(get_srb_thrust(t_sim, cfg.ctrl.F_max, cfg.motor.srb_profile), 28.0)
        if axis == "yaw":
            base = float(np.clip(cfg.pid.Kp_att * (-x[8]) + cfg.pid.Kd_att * (-x[9]), -cfg.phys.delta_max, cfg.phys.delta_max))
            delta = float(np.clip(base + rng.uniform(-0.08, 0.08), -cfg.phys.delta_max, cfg.phys.delta_max))
            U[k, 0] = delta
            wind_vx = rng.normal(0.0, 0.20)
            wind_vy = rng.normal(0.0, 0.55)
            x_next = plant.step(x, np.array([F, 0.0, delta]), dt, wind_vx, wind_vy)
            x_next[10] = max(x_next[10], cfg.phys.m_dry)
            if abs(x_next[8]) > 0.22 or abs(x_next[4]) > 10.0:
                x = np.array([
                    rng.uniform(15.0, 45.0), rng.uniform(18.0, 40.0),
                    0.0, 0.0,
                    rng.uniform(-4.0, 4.0), rng.uniform(-1.0, 1.0),
                    0.0, 0.0,
                    rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03),
                    rng.uniform(cfg.phys.m_dry + 0.4, cfg.phys.m_dry + cfg.phys.m_fuel0),
                ], dtype=float)
            else:
                x = x_next
            X_lat[k + 1] = np.array([x[4], x[5], x[8], x[9]])
        else:
            base = float(np.clip(cfg.pid.Kp_att * (-x[6]) + cfg.pid.Kd_att * (-x[7]), -cfg.phys.delta_max, cfg.phys.delta_max))
            delta = float(np.clip(base + rng.uniform(-0.08, 0.08), -cfg.phys.delta_max, cfg.phys.delta_max))
            U[k, 0] = delta
            wind_vx = rng.normal(0.0, 0.55)
            wind_vy = rng.normal(0.0, 0.20)
            x_next = plant.step(x, np.array([F, delta, 0.0]), dt, wind_vx, wind_vy)
            x_next[10] = max(x_next[10], cfg.phys.m_dry)
            if abs(x_next[6]) > 0.22 or abs(x_next[2]) > 10.0:
                x = np.array([
                    rng.uniform(15.0, 45.0), rng.uniform(18.0, 40.0),
                    rng.uniform(-4.0, 4.0), rng.uniform(-1.0, 1.0),
                    0.0, 0.0,
                    rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03),
                    0.0, 0.0,
                    rng.uniform(cfg.phys.m_dry + 0.4, cfg.phys.m_dry + cfg.phys.m_fuel0),
                ], dtype=float)
            else:
                x = x_next
            X_lat[k + 1] = np.array([x[2], x[3], x[6], x[7]])

    Z_lat = lift_lat.lift(X_lat[:-1])
    Zp_lat = lift_lat.lift(X_lat[1:])
    return X_lat[:-1], U, X_lat[1:], Z_lat, Zp_lat


def run_closed_loop(cfg, controller_type="MPC", waypoint=None):
    """
    waypoint:
      None           -> straight-up benchmark comparison
      [h_pass, x, y] -> steer to (x North, y East) at altitude h_pass
    Both controllers use the same SRB thrust profile.
    """
    run_seed = 0 if cfg.sim.seed is None else int(cfg.sim.seed)
    rng = np.random.default_rng(run_seed)
    rng_train = np.random.default_rng(2026)
    rng_learning = np.random.default_rng(run_seed + 7919)
    model_plant = Rocket3D(cfg.phys)
    actual_phys = copy.deepcopy(cfg.phys)
    if cfg.sim.plant_mismatch_enabled:
        actual_phys.Cd_lat *= cfg.sim.mismatch_cd_lat_scale
        actual_phys.Izz *= cfg.sim.mismatch_inertia_scale
        actual_phys.L_tvc *= cfg.sim.mismatch_tvc_scale
    plant = Rocket3D(actual_phys)

    lift_alt = LiftingMap(cfg.koop.degree, cfg.koop.include_bias)
    X3_tr, F_tr, _, Z3_tr, Z3p_tr = gen_training_data(model_plant, lift_alt, rng_train, cfg.sim.dt, cfg.sim.T_train, cfg)
    edmd_alt = EDMDc(cfg.koop.ridge_lambda)
    A_alt, B_alt = edmd_alt.fit(Z3_tr, F_tr, Z3p_tr)
    C_alt = fit_C(Z3_tr, X3_tr)

    lift_lat = LiftingMap(cfg.koop.degree, cfg.koop.include_bias)
    Xyaw_tr, Dyaw_tr, _, Zyaw_tr, Zyawp_tr = gen_lateral_training_data(model_plant, lift_lat, rng_train, cfg.sim.dt, cfg.sim.T_train, cfg, axis="yaw")
    Xpit_tr, Dpit_tr, _, Zpit_tr, Zpitp_tr = gen_lateral_training_data(model_plant, lift_lat, rng_train, cfg.sim.dt, cfg.sim.T_train, cfg, axis="pitch")
    edmd_yaw = EDMDc(cfg.koop.ridge_lambda)
    edmd_pitch = EDMDc(cfg.koop.ridge_lambda)
    A_yaw, B_yaw = edmd_yaw.fit(Zyaw_tr, Dyaw_tr, Zyawp_tr)
    A_pitch, B_pitch = edmd_pitch.fit(Zpit_tr, Dpit_tr, Zpitp_tr)
    C_yaw = fit_C(Zyaw_tr, Xyaw_tr)
    C_pitch = fit_C(Zpit_tr, Xpit_tr)

    tune = MPC_TUNING
    Q_lat = np.diag([
        cfg.ctrl.Q_y * tune["q_y_scale"],
        cfg.ctrl.Q_vy * tune["q_vy_scale"],
        max(cfg.ctrl.Q_theta, tune["q_theta_min"]),
        tune["q_omega"],
    ])
    R_lat = np.array([[max(cfg.ctrl.R_delta * tune["r_delta_scale"], tune["r_delta_min"])]])
    u_d_min = np.array([-cfg.phys.delta_max])
    u_d_max = np.array([cfg.phys.delta_max])
    lat_horizon = max(cfg.ctrl.horizon, int(tune["lat_horizon"]))
    mpc_yaw = LinearMPC(A_yaw, B_yaw, C_yaw, lat_horizon, u_d_min, u_d_max, Q_lat, R_lat)
    mpc_pitch = LinearMPC(A_pitch, B_pitch, C_pitch, lat_horizon, u_d_min, u_d_max, Q_lat, R_lat)

    rnn = buf = None
    rnn_pitch = rnn_yaw = None
    buf_pitch = buf_yaw = None
    if cfg.rnn.enabled:
        rnn = TinyMLP(dim_in=4, dim_hidden=cfg.rnn.hidden, dim_out=3, lr=cfg.rnn.lr)
        buf = ReplayBuffer(dim_in=4, dim_out=3, maxlen=cfg.rnn.buffer_size)
        rnn_pitch = TinyMLP(dim_in=7, dim_hidden=cfg.rnn.hidden, dim_out=2, lr=cfg.rnn.lr)
        rnn_yaw = TinyMLP(dim_in=7, dim_hidden=cfg.rnn.hidden, dim_out=2, lr=cfg.rnn.lr)
        buf_pitch = ReplayBuffer(dim_in=7, dim_out=2, maxlen=cfg.rnn.buffer_size)
        buf_yaw = ReplayBuffer(dim_in=7, dim_out=2, maxlen=cfg.rnn.buffer_size)

    def lateral_features(state_lat, delta, wind_component, thrust):
        scales = np.array([25.0, 10.0, 0.20, 2.0, cfg.phys.delta_max, 10.0, cfg.ctrl.F_max])
        values = np.array([
            state_lat[0], state_lat[1], state_lat[2], state_lat[3],
            delta, wind_component, thrust,
        ], dtype=float)
        return values / np.maximum(scales, 1e-6)

    def residual_horizon_offset(residual, horizon):
        one_step_limits = np.array([0.0, 0.025, 0.002, 0.015], dtype=float)
        bounded = np.clip(np.asarray(residual, dtype=float), -one_step_limits, one_step_limits)
        blocks = [np.zeros(4, dtype=float)]
        for step_index in range(1, horizon + 1):
            position_error = 0.5 * step_index * (step_index - 1) * cfg.sim.dt * bounded[1]
            velocity_error = step_index * bounded[1]
            angle_error = (
                step_index * bounded[2]
                + 0.5 * step_index * (step_index - 1) * cfg.sim.dt * bounded[3]
            )
            rate_error = step_index * bounded[3]
            blocks.append(np.array([
                position_error, velocity_error, angle_error, rate_error
            ]))
        return np.concatenate(blocks)

    def safe_online_model(A_old, B_old, C_old, A_new, B_new, C_new, Z, U, Z_next, X_next, blend):
        if not all(np.all(np.isfinite(matrix)) for matrix in (A_new, B_new, C_new)):
            return A_old, B_old, C_old, False
        old_radius = float(np.max(np.abs(np.linalg.eigvals(A_old))))
        candidate_A = (1.0 - blend) * A_old + blend * A_new
        candidate_B = (1.0 - blend) * B_old + blend * B_new
        candidate_C = (1.0 - blend) * C_old + blend * C_new
        candidate_radius = float(np.max(np.abs(np.linalg.eigvals(candidate_A))))
        old_b_norm = max(float(np.linalg.norm(B_old)), 1e-8)
        gain_ratio = float(np.linalg.norm(candidate_B)) / old_b_norm
        radius_limit = max(1.03, old_radius * 1.01)
        if candidate_radius > radius_limit or not 0.5 <= gain_ratio <= 2.0:
            return A_old, B_old, C_old, False
        old_x_next = (Z @ A_old.T + U @ B_old.T) @ C_old.T
        candidate_x_next = (Z @ candidate_A.T + U @ candidate_B.T) @ candidate_C.T
        old_error = float(np.mean((X_next - old_x_next) ** 2))
        candidate_error = float(np.mean((X_next - candidate_x_next) ** 2))
        if candidate_error >= 0.995 * old_error:
            return A_old, B_old, C_old, False
        return candidate_A, candidate_B, candidate_C, True

    def adapt_input_gain(A, B, C, Z, U, Z_next, blend):
        baseline_x = (Z @ A.T) @ C.T
        input_effect = (U @ B.T) @ C.T
        observed_effect = (Z_next @ C.T) - baseline_x
        excited = np.abs(U[:, 0]) >= np.deg2rad(0.01)
        if np.count_nonzero(excited) < 6:
            return B, False
        predicted_rate_effect = input_effect[excited, 3]
        observed_rate_effect = observed_effect[excited, 3]
        numerator = float(predicted_rate_effect @ observed_rate_effect)
        denominator = float(predicted_rate_effect @ predicted_rate_effect) + 1e-9
        gain = float(np.clip(numerator / denominator, 0.55, 1.45))
        candidate_B = B * ((1.0 - blend) + blend * gain)
        return A, candidate_B, abs(gain - 1.0) > 1e-3

    pid = PIDController(cfg, cfg.sim.dt)

    gimbal_int_p = 0.0
    gimbal_int_y = 0.0
    mpc_dp_prev = None
    mpc_dy_prev = None
    online_disturbance_pitch = np.zeros(4, dtype=float)
    online_disturbance_yaw = np.zeros(4, dtype=float)
    neural_velocity_pitch = 0.0
    neural_velocity_yaw = 0.0

    wp_active = waypoint is not None
    wp_reached = False
    wp_best_dist = float("inf")
    if waypoint is not None:
        wp_h = float(waypoint[0])
        wp_xp = float(waypoint[1])
        wp_yp = float(waypoint[2])
    else:
        wp_h = wp_xp = wp_yp = 0.0

    T_ascent = cfg.sim.T_eval * 0.55
    s_ref, vz_ref_fn = rocket_ascent_reference(T_ascent, apogee=270.0)

    pitch_init = 0.0
    yaw_init = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pitch_init, 0.0, yaw_init, 0.0, cfg.phys.m_dry + cfg.phys.m_fuel0], dtype=float)

    hist = dict(
        t=[], h=[], vz=[], xp=[], vx=[], y=[], vy=[],
        pitch=[], prate=[], theta=[], omega=[], m=[],
        F=[], delta_p=[], delta=[], h_ref=[], vz_ref=[],
        pred_h=[], pred_vz=[], phase=[],
        wind_vx=[], wind_vy=[], wind_azimuth=[]
    )

    online_koop_updates = 0
    online_buf_z = []
    online_buf_u = []
    online_buf_zp = []
    online_buf_x = []
    online_buf_xp = []

    online_buf_yaw_z = []
    online_buf_yaw_u = []
    online_buf_yaw_zp = []
    online_buf_yaw_x = []

    online_buf_pit_z = []
    online_buf_pit_u = []
    online_buf_pit_zp = []
    online_buf_pit_x = []

    wind = WindModel(rng, cfg)
    steps = int(cfg.sim.T_eval / cfg.sim.dt)
    has_lifted = False

    for k in range(steps):
        t = k * cfg.sim.dt
        h_r = s_ref(t)
        vz_r = vz_ref_fn(t)
        wind_vx, wind_vy = wind.step(cfg.sim.dt)

        if x[0] > 2.0:
            has_lifted = True
        if has_lifted and x[1] < -0.5:
            print(f"  Apogee  t={t:.2f}s  h={x[0]:.1f}m  N={x[2]:.1f}m  E={x[4]:.1f}m")
            break
        if has_lifted and x[0] < 0.1 and x[1] < 0:
            break

        F_cmd = get_srb_thrust(t, cfg.ctrl.F_max, cfg.motor.srb_profile)
        x3 = np.array([x[0], x[1], x[10]], dtype=float)
        z3 = lift_alt.lift(x3).ravel().astype(float)

        if controller_type == "MPC" and has_lifted and t >= tune["activate_time"] and F_cmd >= (x[10] * cfg.phys.g0 * tune["thrust_margin"]):
            guide_to_waypoint = False
            if wp_active and not wp_reached:
                wp_dist = float(np.sqrt((x[0] - wp_h) ** 2 + (x[2] - wp_xp) ** 2 + (x[4] - wp_yp) ** 2))
                wp_best_dist = min(wp_best_dist, wp_dist)
                if wp_dist <= 2.0:
                    wp_reached = True
                    gimbal_int_p = 0.0
                    gimbal_int_y = 0.0
                else:
                    guide_to_waypoint = True

            if guide_to_waypoint:
                xp_bias = np.clip(tune["wp_x_bias_base"] + tune["wp_x_bias_wind"] * wind_vx,
                                  tune["wp_x_bias_min"], tune["wp_x_bias_max"])
                yp_bias = np.clip(tune["wp_y_bias_base"] + tune["wp_y_bias_wind"] * wind_vy + tune["wp_y_cross_wind_vx"] * wind_vx,
                                  tune["wp_y_bias_min"], tune["wp_y_bias_max"])
                xp_target = wp_xp + xp_bias
                yp_target = wp_yp + yp_bias
                h_target = wp_h
            else:
                xp_target = 0.0
                yp_target = 0.0
                h_target = h_r

            z_pitch = lift_lat.lift(np.array([[x[2], x[3], x[6], x[7]]], dtype=float)).ravel()
            z_yaw = lift_lat.lift(np.array([[x[4], x[5], x[8], x[9]]], dtype=float)).ravel()
            pitch_ref = np.tile(np.array([xp_target, 0.0, 0.0, 0.0], dtype=float), mpc_pitch.N + 1)
            yaw_ref = np.tile(np.array([yp_target, 0.0, 0.0, 0.0], dtype=float), mpc_yaw.N + 1)

            if mpc_dp_prev is None:
                dp_init = np.zeros(mpc_pitch.N, dtype=float)
                dy_init = np.zeros(mpc_yaw.N, dtype=float)
            else:
                dp_init = np.hstack([mpc_dp_prev[1:], mpc_dp_prev[-1:]])
                dy_init = np.hstack([mpc_dy_prev[1:], mpc_dy_prev[-1:]])

            pitch_offset = None
            yaw_offset = None
            pitch_prediction_error = online_disturbance_pitch.copy()
            yaw_prediction_error = online_disturbance_yaw.copy()
            pitch_neural_error = np.zeros(2, dtype=float)
            yaw_neural_error = np.zeros(2, dtype=float)
            if rnn_pitch is not None and buf_pitch.X.shape[0] >= cfg.rnn.batch_size:
                pitch_features = lateral_features(
                    np.array([x[2], x[3], x[6], x[7]]),
                    0.0 if mpc_dp_prev is None else mpc_dp_prev[0], wind_vx, F_cmd,
                )
                pitch_neural_error = rnn_pitch.predict(pitch_features[None, :]).ravel()
                neural_velocity_pitch = 0.85 * neural_velocity_pitch + 0.15 * pitch_neural_error[1]
            if rnn_yaw is not None and buf_yaw.X.shape[0] >= cfg.rnn.batch_size:
                yaw_features = lateral_features(
                    np.array([x[4], x[5], x[8], x[9]]),
                    0.0 if mpc_dy_prev is None else mpc_dy_prev[0], wind_vy, F_cmd,
                )
                yaw_neural_error = rnn_yaw.predict(yaw_features[None, :]).ravel()
                neural_velocity_yaw = 0.85 * neural_velocity_yaw + 0.15 * yaw_neural_error[1]
            if cfg.koop.online_enabled:
                pitch_offset = residual_horizon_offset(pitch_prediction_error, mpc_pitch.N)
                yaw_offset = residual_horizon_offset(yaw_prediction_error, mpc_yaw.N)

            dp_seq = mpc_pitch.solve(
                z_pitch, pitch_ref, u_init=dp_init, iters=90, output_offset=pitch_offset
            )
            dy_seq = mpc_yaw.solve(
                z_yaw, yaw_ref, u_init=dy_init, iters=90, output_offset=yaw_offset
            )
            mpc_dp_prev = dp_seq.copy()
            mpc_dy_prev = dy_seq.copy()

            dp_model_cap = np.deg2rad(6.0 if guide_to_waypoint else 3.0)
            dy_model_cap = np.deg2rad(9.0 if guide_to_waypoint else 4.0)
            dp_model = float(np.clip(dp_seq[0], -dp_model_cap, dp_model_cap))
            dy_model = float(np.clip(dy_seq[0], -dy_model_cap, dy_model_cap))

            pd_scale = 0.20 if cfg.sim.isolate_learning_ablation else 1.0
            delta_p_cmd = pd_scale * (cfg.pid.Kp_att * (-x[6]) + cfg.pid.Kd_att * (-x[7]))
            delta_y_cmd = pd_scale * (cfg.pid.Kp_att * (-x[8]) + cfg.pid.Kd_att * (-x[9]))

            if guide_to_waypoint:
                tilt_total = float(np.sqrt(x[6] ** 2 + x[8] ** 2))
                tilt_soft = np.deg2rad(tune["tilt_soft_deg"])
                tilt_hard = np.deg2rad(tune["tilt_hard_deg"])
                if tilt_total <= tilt_soft:
                    tilt_scale = 1.0
                elif tilt_total >= tilt_hard:
                    tilt_scale = 0.45
                else:
                    frac = (tilt_total - tilt_soft) / max(tilt_hard - tilt_soft, 1e-9)
                    tilt_scale = 1.0 - 0.55 * frac
                if tilt_total > np.deg2rad(tune["tilt_cut_deg"]):
                    tilt_scale *= tune["tilt_cut_scale"]

                h_prog = max(h_target, 1.0)
                progress = float(np.clip(x[0] / h_prog, 0.0, 1.0))
                terminal_w = float(np.clip((progress - tune["terminal_start"]) / tune["terminal_span"], 0.0, 1.0))
                lead = progress ** tune["lead_exp"]
                dh = max(h_target - x[0], 0.0)
                v_up = max(abs(x[1]), 0.85)
                t_rem = float(np.clip(dh / v_up, tune["t_rem_min"], tune["t_rem_max"]))
                x_intercept = xp_target - x[3] * t_rem
                y_intercept = yp_target - x[5] * t_rem
                x_des = (1.0 - terminal_w) * (xp_target * lead) + terminal_w * x_intercept
                y_des = (1.0 - terminal_w) * (yp_target * lead) + terminal_w * y_intercept
                ex = float(x_des - x[2])
                ey = float(y_des - x[4])

                F_eff = max(float(F_cmd), 1.0)
                ex_final = float(xp_target - x[2])
                ey_final = float(yp_target - x[4])
                a_ff_x = float(2.0 * (ex_final - x[3] * t_rem) / max(t_rem * t_rem, 1e-6))
                a_ff_y = float(2.0 * (ey_final - x[5] * t_rem) / max(t_rem * t_rem, 1e-6))
                vrel_x = x[3] - wind_vx
                vrel_y = x[5] - wind_vy
                Dwx = 0.5 * cfg.phys.rho * vrel_x * abs(vrel_x) * cfg.phys.Cd_lat * cfg.phys.A_lat
                Dwy = 0.5 * cfg.phys.rho * vrel_y * abs(vrel_y) * cfg.phys.Cd_lat * cfg.phys.A_lat
                delta_cap = np.deg2rad(tune["delta_cap_deg"])
                a_lat_max = float(max(tune["a_lat_min"], tune["a_lat_scale"] * (F_eff / max(x[10], 0.2)) * np.sin(delta_cap)))
                ff_w = tune["ff_base"] + tune["ff_terminal"] * terminal_w
                a_x = float(np.clip(tune["a_x_pos"] * ex + tune["a_x_vel"] * (-x[3]) + ff_w * a_ff_x, -a_lat_max, a_lat_max))
                a_y = float(np.clip(tune["a_y_pos"] * ey + tune["a_y_vel"] * (-x[5]) + ff_w * a_ff_y, -a_lat_max, a_lat_max))
                if terminal_w > tune["overshoot_terminal"] and ey < 0.0:
                    a_y = float(np.clip(tune["overshoot_scale"] * a_y - tune["overshoot_bias"] * abs(ey), -a_lat_max, a_lat_max))
                dp_ff = float(np.clip(-(x[10] / F_eff) * a_x * tilt_scale, -delta_cap, delta_cap))
                dy_ff = float(np.clip(-(x[10] / F_eff) * a_y * tilt_scale, -delta_cap, delta_cap))
                dp_wind_ff = float(np.clip(-tune["dp_wind_ff_scale"] * np.arcsin(np.clip(Dwx / F_eff, -0.12, 0.12)), -np.deg2rad(tune["dp_wind_ff_cap_deg"]), np.deg2rad(tune["dp_wind_ff_cap_deg"])))
                dy_wind_ff = float(np.clip(-tune["dy_wind_ff_scale"] * np.arcsin(np.clip(Dwy / F_eff, -0.12, 0.12)), -np.deg2rad(tune["dy_wind_ff_cap_deg"]), np.deg2rad(tune["dy_wind_ff_cap_deg"])))
                east_scale = float(np.clip(1.0 - tune["east_scale_wind_gain"] * max(wind_vy, 0.0), tune["east_scale_min"], 1.0))

                gimbal_int_p += tune["waypoint_int_p"] * tilt_scale * ex * cfg.sim.dt
                gimbal_int_y += tune["waypoint_int_y"] * east_scale * tilt_scale * ey * cfg.sim.dt
                gimbal_int_p = float(np.clip(gimbal_int_p, -tune["waypoint_int_cap_p"], tune["waypoint_int_cap_p"]))
                gimbal_int_y = float(np.clip(gimbal_int_y, -tune["waypoint_int_cap_y"], tune["waypoint_int_cap_y"]))

                delta_p_cmd = float(np.clip(delta_p_cmd + tune["waypoint_model_p"] * dp_model + tune["waypoint_ff_p"] * dp_ff + dp_wind_ff + gimbal_int_p, -cfg.phys.delta_max, cfg.phys.delta_max))
                delta_y_cmd = float(np.clip(delta_y_cmd + tune["waypoint_model_y"] * dy_model + east_scale * (tune["waypoint_ff_y"] * dy_ff + dy_wind_ff + gimbal_int_y), -cfg.phys.delta_max, cfg.phys.delta_max))
            else:
                vrel_x = x[3] - wind_vx
                vrel_y = x[5] - wind_vy
                Dx = 0.5 * cfg.phys.rho * vrel_x * abs(vrel_x) * cfg.phys.Cd_lat * cfg.phys.A_lat
                Dy = 0.5 * cfg.phys.rho * vrel_y * abs(vrel_y) * cfg.phys.Cd_lat * cfg.phys.A_lat
                F_eff = max(float(F_cmd), 1.0)
                pitch_ff = float(np.clip(-np.arcsin(np.clip(Dx / F_eff, -0.14, 0.14)), -np.deg2rad(6.0), np.deg2rad(6.0)))
                yaw_ff = float(np.clip(-np.arcsin(np.clip(Dy / F_eff, -0.14, 0.14)), -np.deg2rad(6.0), np.deg2rad(6.0)))
                integral_scale = 0.0 if cfg.sim.isolate_learning_ablation else 1.0
                gimbal_int_p = float(np.clip(tune["benchmark_int_decay"] * gimbal_int_p - integral_scale * tune["benchmark_int_pos_gain"] * x[2] * cfg.sim.dt, -tune["benchmark_int_cap"], tune["benchmark_int_cap"]))
                gimbal_int_y = float(np.clip(tune["benchmark_int_decay"] * gimbal_int_y - integral_scale * tune["benchmark_int_pos_gain"] * x[4] * cfg.sim.dt, -tune["benchmark_int_cap"], tune["benchmark_int_cap"]))
                ff_scale = 0.0 if cfg.sim.isolate_learning_ablation else 1.0
                model_scale = 2.5 if cfg.sim.isolate_learning_ablation else 1.0
                delta_p_cmd = float(np.clip(delta_p_cmd + model_scale * tune["benchmark_model_p"] * dp_model + ff_scale * tune["benchmark_ff_p"] * pitch_ff + gimbal_int_p, -cfg.phys.delta_max, cfg.phys.delta_max))
                delta_y_cmd = float(np.clip(delta_y_cmd + model_scale * tune["benchmark_model_y"] * dy_model + ff_scale * tune["benchmark_ff_y"] * yaw_ff + gimbal_int_y, -cfg.phys.delta_max, cfg.phys.delta_max))

            online_gain = 0.65 if cfg.sim.isolate_learning_ablation else 0.20
            residual_gain = 0.12 if cfg.sim.isolate_learning_ablation else 0.04
            disturbance_cap = np.deg2rad(3.0)
            residual_cap = np.deg2rad(1.0)
            F_eff = max(float(F_cmd), 1.0)
            pitch_disturbance_trim = np.clip(
                (x[10] / F_eff)
                * online_gain * online_disturbance_pitch[1]
                / cfg.sim.dt,
                -disturbance_cap, disturbance_cap,
            )
            yaw_disturbance_trim = np.clip(
                (x[10] / F_eff)
                * online_gain * online_disturbance_yaw[1]
                / cfg.sim.dt,
                -disturbance_cap, disturbance_cap,
            )
            pitch_residual_trim = np.clip(
                residual_gain * (x[10] / F_eff) * neural_velocity_pitch / cfg.sim.dt,
                -residual_cap, residual_cap,
            )
            yaw_residual_trim = np.clip(
                residual_gain * (x[10] / F_eff) * neural_velocity_yaw / cfg.sim.dt,
                -residual_cap, residual_cap,
            )
            delta_p_cmd = float(np.clip(
                delta_p_cmd + pitch_disturbance_trim + pitch_residual_trim,
                -cfg.phys.delta_max, cfg.phys.delta_max,
            ))
            delta_y_cmd = float(np.clip(
                delta_y_cmd + yaw_disturbance_trim + yaw_residual_trim,
                -cfg.phys.delta_max, cfg.phys.delta_max,
            ))

        else:
            delta_p_cmd, delta_y_cmd = compute_attitude_delta_pid(x, cfg)
            gimbal_int_p *= 0.96
            gimbal_int_y *= 0.96
            mpc_dp_prev = None
            mpc_dy_prev = None

        gimbal_noise_std = tune["gimbal_noise_mpc"] if controller_type == "MPC" else 0.0020
        F_noisy = float(F_cmd)
        delta_p_noisy = float(np.clip(delta_p_cmd + rng.normal(0.0, gimbal_noise_std), -cfg.phys.delta_max, cfg.phys.delta_max))
        delta_y_noisy = float(np.clip(delta_y_cmd + rng.normal(0.0, gimbal_noise_std), -cfg.phys.delta_max, cfg.phys.delta_max))
        u_actual = np.array([F_noisy, delta_p_noisy, delta_y_noisy])

        z3_pred = A_alt @ z3 + B_alt[:, 0] * F_noisy
        x3_pred = (C_alt @ z3_pred).ravel()
        if rnn is not None:
            x3_pred = x3_pred + rnn.predict(np.array([[x[0], x[1], x[10], F_noisy]])).ravel()

        mismatch_active = cfg.sim.plant_mismatch_enabled and t >= cfg.sim.mismatch_start_time
        active_plant = plant if mismatch_active else model_plant
        x_next = active_plant.step(x, u_actual, cfg.sim.dt, wind_vx, wind_vy)
        if mismatch_active and F_noisy > 0.0:
            x_next[3] += cfg.sim.dt * cfg.sim.mismatch_accel_north
            x_next[5] += cfg.sim.dt * cfg.sim.mismatch_accel_east

        if rnn_pitch is not None:
            x_pitch_now = np.array([x[2], x[3], x[6], x[7]], dtype=float)
            x_pitch_next = np.array([x_next[2], x_next[3], x_next[6], x_next[7]], dtype=float)
            z_pitch_now = lift_lat.lift(x_pitch_now).ravel()
            x_pitch_koop = C_pitch @ (A_pitch @ z_pitch_now + B_pitch[:, 0] * delta_p_noisy)
            pitch_model_error = x_pitch_next - x_pitch_koop
            feat_pitch = lateral_features(x_pitch_now, delta_p_noisy, wind_vx, F_noisy)
            buf_pitch.push(feat_pitch, pitch_model_error[:2] - online_disturbance_pitch[:2])

            x_yaw_now_r = np.array([x[4], x[5], x[8], x[9]], dtype=float)
            x_yaw_next_r = np.array([x_next[4], x_next[5], x_next[8], x_next[9]], dtype=float)
            z_yaw_now_r = lift_lat.lift(x_yaw_now_r).ravel()
            x_yaw_koop = C_yaw @ (A_yaw @ z_yaw_now_r + B_yaw[:, 0] * delta_y_noisy)
            yaw_model_error = x_yaw_next_r - x_yaw_koop
            feat_yaw = lateral_features(x_yaw_now_r, delta_y_noisy, wind_vy, F_noisy)
            buf_yaw.push(feat_yaw, yaw_model_error[:2] - online_disturbance_yaw[:2])

            if buf_pitch.X.shape[0] >= cfg.rnn.batch_size:
                idx_pitch = rng_learning.choice(buf_pitch.X.shape[0], cfg.rnn.batch_size, replace=False)
                idx_yaw = rng_learning.choice(buf_yaw.X.shape[0], cfg.rnn.batch_size, replace=False)
                for _ in range(max(cfg.rnn.epochs_per_update, 6)):
                    rnn_pitch.train_batch(buf_pitch.X[idx_pitch], buf_pitch.Y[idx_pitch])
                    rnn_yaw.train_batch(buf_yaw.X[idx_yaw], buf_yaw.Y[idx_yaw])

        if cfg.koop.online_enabled:
            x_pitch_now_obs = np.array([x[2], x[3], x[6], x[7]], dtype=float)
            x_pitch_next_obs = np.array([x_next[2], x_next[3], x_next[6], x_next[7]], dtype=float)
            z_pitch_obs = lift_lat.lift(x_pitch_now_obs).ravel()
            pitch_observed_error = x_pitch_next_obs - C_pitch @ (
                A_pitch @ z_pitch_obs + B_pitch[:, 0] * delta_p_noisy
            )
            x_yaw_now_obs = np.array([x[4], x[5], x[8], x[9]], dtype=float)
            x_yaw_next_obs = np.array([x_next[4], x_next[5], x_next[8], x_next[9]], dtype=float)
            z_yaw_obs = lift_lat.lift(x_yaw_now_obs).ravel()
            yaw_observed_error = x_yaw_next_obs - C_yaw @ (
                A_yaw @ z_yaw_obs + B_yaw[:, 0] * delta_y_noisy
            )
            observer_gain = 0.25 if cfg.sim.isolate_learning_ablation else 0.08
            online_disturbance_pitch[1] = (
                (1.0 - observer_gain) * online_disturbance_pitch[1]
                + observer_gain * pitch_observed_error[1]
            )
            online_disturbance_yaw[1] = (
                (1.0 - observer_gain) * online_disturbance_yaw[1]
                + observer_gain * yaw_observed_error[1]
            )

        if cfg.sim.meas_noise_std > 0.0 and rnn is not None and k % cfg.rnn.train_every == 0:
            x3_actual = np.array([x_next[0], x_next[1], x_next[10]])
            z3c = lift_alt.lift(np.array([x[0], x[1], x[10]])).ravel()
            x3_koop = (C_alt @ (A_alt @ z3c + B_alt[:, 0] * F_noisy)).ravel()
            buf.push(np.array([x[0], x[1], x[10], F_noisy]), x3_actual - x3_koop)
            if buf.X.shape[0] >= cfg.rnn.batch_size:
                idx = rng_learning.choice(buf.X.shape[0], cfg.rnn.batch_size, replace=False)
                for _ in range(cfg.rnn.epochs_per_update):
                    rnn.train_batch(buf.X[idx], buf.Y[idx])

        if cfg.koop.online_enabled:
            x3_now = np.array([x[0], x[1], x[10]], dtype=float)
            x3_next = np.array([x_next[0], x_next[1], x_next[10]], dtype=float)
            z3_now = z3.copy()
            z3_next = lift_alt.lift(x3_next).ravel().astype(float)
            online_buf_z.append(z3_now)
            online_buf_u.append(np.array([F_noisy], dtype=float))
            online_buf_zp.append(z3_next)
            online_buf_x.append(x3_now)
            online_buf_xp.append(x3_next)

            # buffer lateral states for pitch and yaw online identification
            x_yaw_now = np.array([x[4], x[5], x[8], x[9]], dtype=float)
            x_yaw_next = np.array([x_next[4], x_next[5], x_next[8], x_next[9]], dtype=float)
            z_yaw_now = lift_lat.lift(x_yaw_now).ravel().astype(float)
            z_yaw_next = lift_lat.lift(x_yaw_next).ravel().astype(float)
            online_buf_yaw_z.append(z_yaw_now)
            online_buf_yaw_u.append(np.array([delta_y_noisy], dtype=float))
            online_buf_yaw_zp.append(z_yaw_next)
            online_buf_yaw_x.append(x_yaw_now)

            x_pit_now = np.array([x[2], x[3], x[6], x[7]], dtype=float)
            x_pit_next = np.array([x_next[2], x_next[3], x_next[6], x_next[7]], dtype=float)
            z_pit_now = lift_lat.lift(x_pit_now).ravel().astype(float)
            z_pit_next = lift_lat.lift(x_pit_next).ravel().astype(float)
            online_buf_pit_z.append(z_pit_now)
            online_buf_pit_u.append(np.array([delta_p_noisy], dtype=float))
            online_buf_pit_zp.append(z_pit_next)
            online_buf_pit_x.append(x_pit_now)

            if len(online_buf_z) > cfg.koop.online_window:
                online_buf_z.pop(0)
                online_buf_u.pop(0)
                online_buf_zp.pop(0)
                online_buf_x.pop(0)
                online_buf_xp.pop(0)

                online_buf_yaw_z.pop(0)
                online_buf_yaw_u.pop(0)
                online_buf_yaw_zp.pop(0)
                online_buf_yaw_x.pop(0)

                online_buf_pit_z.pop(0)
                online_buf_pit_u.pop(0)
                online_buf_pit_zp.pop(0)
                online_buf_pit_x.pop(0)

            min_samples = max(int(cfg.koop.online_min_samples), z_yaw_now.shape[0] + 4)
            can_update = (
                len(online_buf_z) >= min_samples
                and (k + 1) % max(int(cfg.koop.online_update_every), 1) == 0
            )
            if can_update:
                Z_on = np.vstack(online_buf_z)
                U_on = np.vstack(online_buf_u)
                Zp_on = np.vstack(online_buf_zp)
                X_on = np.vstack(online_buf_x)

                Zy_on = np.vstack(online_buf_yaw_z)
                Uy_on = np.vstack(online_buf_yaw_u)
                Zpy_on = np.vstack(online_buf_yaw_zp)
                Xy_on = np.vstack(online_buf_yaw_x)

                Zp_pit_on = np.vstack(online_buf_pit_z)
                Up_pit_on = np.vstack(online_buf_pit_u)
                Zpp_pit_on = np.vstack(online_buf_pit_zp)
                Xp_pit_on = np.vstack(online_buf_pit_x)

                try:
                    A_new, B_new = EDMDc(cfg.koop.ridge_lambda).fit(Z_on, U_on, Zp_on)
                    C_new = fit_C(Z_on, X_on)

                    blend = float(np.clip(cfg.koop.online_blend, 0.0, 1.0))
                    A_alt, B_alt, C_alt, _ = safe_online_model(
                        A_alt, B_alt, C_alt, A_new, B_new, C_new,
                        Z_on, U_on, Zp_on, np.vstack(online_buf_xp), blend,
                    )

                    A_yaw, B_yaw, yaw_accepted = adapt_input_gain(
                        A_yaw, B_yaw, C_yaw, Zy_on, Uy_on, Zpy_on, blend
                    )
                    A_pitch, B_pitch, pitch_accepted = adapt_input_gain(
                        A_pitch, B_pitch, C_pitch, Zp_pit_on, Up_pit_on, Zpp_pit_on, blend
                    )

                    # update live MPC predictive model matrices
                    if yaw_accepted:
                        mpc_yaw = LinearMPC(A_yaw, B_yaw, C_yaw, lat_horizon, u_d_min, u_d_max, Q_lat, R_lat)
                    if pitch_accepted:
                        mpc_pitch = LinearMPC(A_pitch, B_pitch, C_pitch, lat_horizon, u_d_min, u_d_max, Q_lat, R_lat)

                    online_koop_updates += int(yaw_accepted or pitch_accepted)
                except np.linalg.LinAlgError:
                    pass

        phase = (
            "Pre-Launch" if not has_lifted else
            "Ascent" if x[1] > 1.5 else
            "Apogee" if x[1] >= -0.5 else "Descent"
        )

        hist["t"].append(t)
        hist["h"].append(float(x[0]))
        hist["vz"].append(float(x[1]))
        hist["xp"].append(float(x[2]))
        hist["vx"].append(float(x[3]))
        hist["y"].append(float(x[4]))
        hist["vy"].append(float(x[5]))
        hist["pitch"].append(float(x[6]))
        hist["prate"].append(float(x[7]))
        hist["theta"].append(float(x[8]))
        hist["omega"].append(float(x[9]))
        hist["m"].append(float(x[10]))
        hist["F"].append(float(u_actual[0]))
        hist["delta_p"].append(float(u_actual[1]))
        hist["delta"].append(float(u_actual[2]))
        hist["h_ref"].append(float(h_r))
        hist["vz_ref"].append(float(vz_r))
        hist["pred_h"].append(float(x3_pred[0]))
        hist["pred_vz"].append(float(x3_pred[1]))
        hist["phase"].append(phase)
        hist["wind_vx"].append(float(wind_vx))
        hist["wind_vy"].append(float(wind_vy))
        hist["wind_azimuth"].append(float(wind.azimuth))
        x = x_next

    if not hist["h"]:
        raise ValueError("Simulation produced no data.")

    hist["controller"] = controller_type
    hist["online_koop_updates"] = online_koop_updates
    hist["residual_buffer_size"] = int(buf.X.shape[0]) if buf is not None else 0
    return hist, {"A": A_alt, "B": B_alt, "C": C_alt, "online_koop_updates": online_koop_updates}


def run_monte_carlo(cfg, controller_type="MPC", waypoint=None, shared_base_seed=None):
    """Run N_MC_runs simulations with varying random seeds."""
    all_hist = []
    mats = None
    if shared_base_seed is not None:
        base_seed = shared_base_seed
    elif cfg.sim.seed is None:
        base_seed = int(_time.time() * 1000) % 999983
    else:
        base_seed = int(cfg.sim.seed)

    for i in range(cfg.sim.N_MC_runs):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.sim.seed = base_seed + i * 7
        print(f"  Run {i+1}/{cfg.sim.N_MC_runs}  (seed={cfg_i.sim.seed})")
        hist, mats = run_closed_loop(cfg_i, controller_type, waypoint)
        all_hist.append(hist)
    return all_hist, mats


def make_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
