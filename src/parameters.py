from dataclasses import dataclass, field

@dataclass
class PhysParams:
    g0: float = 9.80665        #gravity [m/s^2]
    rho: float = 1.225         #sea-level air density [kg/m^3]
    Cd: float = 0.45           #axial drag coefficient
    Cd_lat: float = 1.1        #lateral drag coefficient
    A_ref: float = 0.00785     #axial area [m^2]
    A_lat: float = 0.04        #lateral area [m^2]
    Isp: float = 210.0         #specific impulse [s]
    m_dry: float = 2.0         #dry mass [kg]
    m_fuel0: float = 1.5       #initial fuel mass [kg]
    Izz: float = 0.08          #inertia around pitch/yaw [kg*m^2]
    L_tvc: float = 0.30        #tvc lever arm [m]
    L_cp: float = -0.001       #small passive stability offset
    delta_max: float = 0.26    #max gimbal angle [rad]

@dataclass
class MotorParams:
    motor_type: str = "srb"
    srb_profile: tuple = (
        (0.00, 0.00),
        (0.05, 1.00),
        (0.30, 0.96),
        (4.00, 0.92),
        (7.50, 0.85),
        (9.00, 0.60),
        (9.80, 0.15),
        (10.0, 0.00),
    )

@dataclass
class SimParams:
    dt: float = 0.02
    T_train: float = 15.0
    T_eval: float = 25.0
    seed: int = None
    wind_speed_ms: float = 5.5
    wind_gust_factor: float = 2.2
    wind_gust_prob: float = 0.012
    meas_noise_std: float = 0.01
    thrust_std: float = 0.015
    N_MC_runs: int = 5
    plant_mismatch_enabled: bool = False
    mismatch_start_time: float = 0.0
    mismatch_cd_lat_scale: float = 1.0
    mismatch_inertia_scale: float = 1.0
    mismatch_tvc_scale: float = 1.0
    mismatch_accel_north: float = 0.0
    mismatch_accel_east: float = 0.0
    isolate_learning_ablation: bool = False

@dataclass
class PIDParams:
    Kp_alt: float = 3.5
    Ki_alt: float = 0.03
    Kd_alt: float = 1.5
    Kp_att: float = 2.0
    Ki_att: float = 0.0
    Kd_att: float = 0.22
    Kp_lat: float = 0.025
    Ki_lat: float = 0.003
    Kd_lat: float = 0.08

@dataclass
class ControlParams:
    F_min: float = 0.0
    F_max: float = 55.0
    delta_max: float = 0.26
    horizon: int = 20
    Q_h: float = 120.0
    Q_vz: float = 2.0
    Q_m: float = 0.01
    R_F: float = 0.25
    Q_y: float = 120.0
    Q_vy: float = 20.0
    Q_theta: float = 8.0
    Q_omega: float = 0.0
    R_delta: float = 5.0

@dataclass
class KoopmanParams:
    degree: int = 2
    ridge_lambda: float = 1e-4
    online_window: int = 800
    include_bias: bool = True
    online_enabled: bool = True
    online_update_every: int = 8
    online_blend: float = 0.08
    online_min_samples: int = 120

@dataclass
class ResidualNNParams:
    enabled: bool = True
    hidden: int = 32
    lr: float = 8e-4
    batch_size: int = 64
    train_every: int = 5
    epochs_per_update: int = 3
    buffer_size: int = 8000

@dataclass
class OutputParams:
    outdir: str = "artifacts"

@dataclass
class Config:
    phys: PhysParams = field(default_factory=PhysParams)
    motor: MotorParams = field(default_factory=MotorParams)
    sim: SimParams = field(default_factory=SimParams)
    ctrl: ControlParams = field(default_factory=ControlParams)
    koop: KoopmanParams = field(default_factory=KoopmanParams)
    rnn: ResidualNNParams = field(default_factory=ResidualNNParams)
    pid: PIDParams = field(default_factory=PIDParams)
    out: OutputParams = field(default_factory=OutputParams)

cfg = Config()
