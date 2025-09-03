#Hunter Weston

#Params

from dataclasses import dataclass

@dataclass
class PhysParams:
    g0: float = 9.80665       #g [m/s^2]
    rho: float = 1.225        #air density [kg/m^3]
    Cd: float = 0.5           #drag coefficent
    A: float = 0.01           #ref area [m^2]
    Isp: float = 250.0        #specific impulse [s]
    m_dry: float = 1.0        #dry mass [kg]
    m_fuel0: float = 1.0      #fuel mass [kg] (assumes full fuel consumption)

@dataclass
class SimParams:
    dt: float = 0.02          #simulation timestep [s]
    T_train: float = 10.0     #training data gen [s]
    T_eval: float = 10.0      #closed loop eval [s]
    seed: int = 42            #RNG seed
    wind_std: float = 0.0     #vert wind disturbance std [m/s^2] (accel noise)
    meas_noise_std: float = 0.0 #measurement noise for h and v [units of state]
    thrust_std: float = 0.05  #fractional Gaussian noise on thrust
    N_MC_runs: int = 5         #Monte Carlo runs / controller

@dataclass
class PIDParams:
    Kp: float = 5.0  #Proprotional
    Ki: float = 0.1  #Integral
    Kd: float = 2.0  #Derivative

@dataclass
class ControlParams:
    u_min: float = 0.0        #min thrust [N]
    u_max: float = 40.0       #max thrust [N]
    horizon: int = 30         #MPC horizon
    Q_h: float = 100.0        #altitude weight
    Q_v: float = 10.0         #velocity weight
    Q_m: float = 1.0          #mass weight (idk why mass has to be part of the cost function but it seems to work)
    R_u: float = 0.01         #control effort weight
    ref_altitude: float = 0.0 #ref landing altitude [m]
    ref_velocity: float = 0.0 #ref landing velocity [m/s]

@dataclass
class KoopmanParams:
    degree: int = 2           #poly lifting degree
    ridge_lambda: float = 1e-4 #ridge regularization for EDMD
    online_window: int = 400  #sliding window size (online updates)
    update_every: int = 20    #refit frequency
    include_bias: bool = True  #seems to work, idk why tho lol

@dataclass
class ResidualNNParams:
    enabled: bool = True
    hidden: int = 16          #hidden units for MLP
    lr: float = 1e-3          #learning rate for SGD
    batch_size: int = 64
    train_every: int = 5      #trains NN every N steps
    epochs_per_update: int = 2
    buffer_size: int = 5000   #replay buffer size

@dataclass
class OutputParams:
    outdir: str = "artifacts"
    make_plots: bool = True
    save_csv: bool = True
    save_npz: bool = True

@dataclass
class Config:
    from dataclasses import field
    phys: PhysParams = field(default_factory=PhysParams)
    sim: SimParams = field(default_factory=SimParams)
    ctrl: ControlParams = field(default_factory=ControlParams)
    koop: KoopmanParams = field(default_factory=KoopmanParams)
    rnn: ResidualNNParams = field(default_factory=ResidualNNParams)
    pid: PIDParams = field(default_factory=PIDParams)
    out: OutputParams = field(default_factory=OutputParams)

cfg = Config()
