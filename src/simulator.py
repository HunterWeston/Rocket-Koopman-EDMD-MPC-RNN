#Hunter Weston

#Generates training data w RNG
#Fits EDMD with control in lifted space
#Fits C for state reconstruction
#Builds MPC, runs closed loop sim
#Online EDMD updates with sliding window
#Learns RNN on r(x,u), corrects Koop predictor

import os, numpy as np, csv
from parameters import cfg
from rocket_model import Rocket1D, rocket_altitude_parabola
from lifting import LiftingMap, fit_C
from edmd import EDMDc, OnlineBuffer
from residual_nn import TinyMLP, ReplayBuffer
from mpc import LinearMPC, build_reference

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, u_min, u_max):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, h, h_ref, v):
        error = h_ref - h
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        u = self.Kp * error + self.Ki * self.integral - self.Kd * v
        return np.clip(u, self.u_min, self.u_max)

def run_monte_carlo(cfg, controller_type="MPC"):
    all_hist = []
    mats = None
    for i in range(cfg.sim.N_MC_runs):
        cfg.sim.seed += i
        hist, mats = run_closed_loop(cfg, controller_type)
        all_hist.append(hist)
    return all_hist, mats


def run_closed_loop(cfg, controller_type="MPC"):
    rng = np.random.default_rng(cfg.sim.seed)
    plant = Rocket1D(cfg.phys)
    lift = LiftingMap(cfg.koop.degree, cfg.koop.include_bias)

    X, U, Xp, Z, Zp = gen_training_data(plant, lift, rng, cfg.sim.dt, cfg.sim.T_train,
                                        cfg.ctrl.u_min, cfg.ctrl.u_max, cfg.sim.wind_std)
    edmd = EDMDc(cfg.koop.ridge_lambda)
    A, B = edmd.fit(Z, U, Zp)
    C = fit_C(Z, X)

    Q = np.diag([cfg.ctrl.Q_h * 20, cfg.ctrl.Q_v, cfg.ctrl.Q_m])
    R = cfg.ctrl.R_u * 0.1
    mpc = LinearMPC(A, B, C, cfg.ctrl.horizon, cfg.ctrl.u_min, cfg.ctrl.u_max, Q, R)

    rnn = None
    buf = None
    if cfg.rnn.enabled:
        rnn = TinyMLP(dim_in=4, dim_hidden=cfg.rnn.hidden, dim_out=3, lr=cfg.rnn.lr)
        buf = ReplayBuffer(dim_in=4, dim_out=3, maxlen=cfg.rnn.buffer_size)

    obuf = OnlineBuffer(dz=Z.shape[1], du=1, maxlen=cfg.koop.online_window)
    for k in range(Z.shape[0]):
        obuf.append(Z[k], U[k], Zp[k])

    pid = None
    if controller_type == "PID":
        pid = PIDController(cfg.pid.Kp, cfg.pid.Ki, cfg.pid.Kd, cfg.sim.dt,
                            cfg.ctrl.u_min, cfg.ctrl.u_max)

    #initial state
    x = np.array([0.0, 32.0, cfg.phys.m_dry + cfg.phys.m_fuel0], dtype=float)
    z = lift.lift(x).ravel()
    z = z.astype(float)
    hist = dict(t=[], h=[], v=[], m=[], u=[], h_ref=[], v_ref=[], pred_h=[], pred_v=[], pred_m=[])
    u_prev_seq = None

    s_ref, v_ref, _ = rocket_altitude_parabola(cfg.sim.T_eval, apogee=80.0)

    steps = int(cfg.sim.T_eval / cfg.sim.dt)

    if s_ref(0) < 0:
        raise ValueError("ref trajectory doesnt work, altitude at t=0 is negative.")

    for k in range(steps):
        t = k * cfg.sim.dt
        ref_alt = s_ref(t)
        ref_vel = v_ref(t)

        if ref_alt < 0:
            print(f"sim stopped at t={t:.2f}s because ref altitude went negative.")
            break

        if controller_type == "PID":
            u_cmd = pid.step(x[0], ref_alt, x[1])
        else:
            ref = build_reference(cfg, cfg.ctrl.horizon)
            ref[::3] = [s_ref(t + i * cfg.sim.dt) for i in range(cfg.ctrl.horizon + 1)]
            ref[1::3] = [v_ref(t + i * cfg.sim.dt) for i in range(cfg.ctrl.horizon + 1)]
            u_seq = mpc.solve(z, ref, u_init=u_prev_seq)
            u_cmd = float(u_seq[0])
            u_prev_seq = np.hstack([u_seq[1:], u_seq[-1]])

        u_actual = u_cmd * (1 + rng.normal(0, cfg.sim.thrust_std))

        z_next_pred = A @ z + (B[:, 0] * u_actual)
        x_next_pred = (C @ z_next_pred).ravel()
        if cfg.rnn.enabled:
            r_in = np.hstack([x, u_actual]).reshape(1, -1)
            r_hat = rnn.predict(r_in).ravel()
            x_next_pred += r_hat

        wind = rng.normal(0, cfg.sim.wind_std)
        x_next = plant.step(x, u_actual, cfg.sim.dt, wind)

        #measurement noise
        meas = x_next.copy()
        if cfg.sim.meas_noise_std>0.0:
            meas[:2] += rng.normal(0, cfg.sim.meas_noise_std, size=2)
        z_next = lift.lift(meas).ravel()

        #online EDMD update
        obuf.append(z, np.array([u_actual]), z_next)
        if (k+1)%cfg.koop.update_every==0:
            A, B = edmd.fit(obuf.Z, obuf.U, obuf.Zp)
            mpc = LinearMPC(A, B, C, cfg.ctrl.horizon, cfg.ctrl.u_min, cfg.ctrl.u_max, Q, R)

        #RNN training
        if cfg.rnn.enabled and (k%cfg.rnn.train_every==0):
            r_target = x_next - (C@(A@z + (B[:,0]*u_actual))).ravel()
            buf.push(np.hstack([x,u_actual]), r_target)
            if buf.X.shape[0]>=cfg.rnn.batch_size:
                idx = np.random.choice(buf.X.shape[0], cfg.rnn.batch_size, replace=False)
                for _ in range(cfg.rnn.epochs_per_update):
                    rnn.train_batch(buf.X[idx], buf.Y[idx])

        hist['t'].append(t)
        hist['h'].append(x[0])
        hist['v'].append(x[1])
        hist['m'].append(x[2])
        hist['u'].append(u_actual)
        hist['h_ref'].append(ref_alt)
        hist['v_ref'].append(ref_vel)
        hist['pred_h'].append(x_next_pred[0])
        hist['pred_v'].append(x_next_pred[1])
        hist['pred_m'].append(x_next_pred[2])

        x = x_next
        z = lift.lift(x).ravel()

    if not hist['h'] or not hist['h_ref']:
        raise ValueError("Simulation produced no valid data. Check initial conditions or controller behavior.")

    return hist, {'A': A, 'B': B, 'C': C}


def make_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)

def gen_training_data(plant, lift, rng, dt, T, u_min, u_max, wind_std=0.0):
    steps = int(T/dt)
    X = np.zeros((steps+1, 3))
    U = np.zeros((steps, 1))
    Z = None
    x = np.array([50.0, -10.0, plant.p.m_dry + plant.p.m_fuel0], dtype=float)
    X[0] = x
    for k in range(steps):
        u = rng.uniform(u_min, u_max)
        wind = rng.normal(0, wind_std)
        x = plant.step(x, u, dt, wind)
        X[k+1] = x
        U[k] = u
    Z = lift.lift(X[:-1, :])
    Zp = lift.lift(X[1:, :])
    return X[:-1, :], U, X[1:, :], Z, Zp

def build_Q_matrix(cfg):
    Q = np.diag([cfg.ctrl.Q_h, cfg.ctrl.Q_v, cfg.ctrl.Q_m])
    return Q


def save_artifacts(cfg, hist, mats):
    import numpy as np
    outdir = os.path.join(cfg.out.outdir)
    make_dirs(outdir)

    # CSV
    if cfg.out.save_csv:
        csv_path = os.path.join(outdir, "data", "timeseries.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            keys = ['t','h','v','m','u','h_ref','v_ref','pred_h','pred_v','pred_m']
            w.writerow(keys)
            for i in range(len(hist['t'])):
                w.writerow([hist[k][i] for k in keys])

    # NPZ
    if cfg.out.save_npz:
        npz_path = os.path.join(outdir, "data", "matrices_and_hist.npz")
        np.savez(npz_path,
                 A=mats['A'], B=mats['B'], C=mats['C'],
                 t=np.array(hist['t']), h=np.array(hist['h']), v=np.array(hist['v']), m=np.array(hist['m']),
                 u=np.array(hist['u']), h_ref=np.array(hist['h_ref']), v_ref=np.array(hist['v_ref']),
                 pred_h=np.array(hist['pred_h']), pred_v=np.array(hist['pred_v']), pred_m=np.array(hist['pred_m']))

    # Plots
    if cfg.out.make_plots:
        import matplotlib.pyplot as plt
        # 1) Altitude vs time
        plt.figure()
        plt.plot(hist['t'], hist['h'], label="h")
        plt.plot(hist['t'], hist['h_ref'], label="h_ref")
        plt.xlabel("time [s]"); plt.ylabel("altitude [m]"); plt.legend(); plt.title("Altitude Tracking")
        p1 = os.path.join(outdir, "plots", "altitude.png"); plt.savefig(p1, dpi=160); plt.close()

        # 2) Velocity vs time
        plt.figure()
        plt.plot(hist['t'], hist['v'], label="v")
        plt.plot(hist['t'], hist['v_ref'], label="v_ref")
        plt.xlabel("time [s]"); plt.ylabel("velocity [m/s]"); plt.legend(); plt.title("Velocity Tracking")
        p2 = os.path.join(outdir, "plots", "velocity.png"); plt.savefig(p2, dpi=160); plt.close()

        # 3) Thrust input
        plt.figure()
        plt.step(hist['t'], hist['u'], where='post', label="u (thrust)")
        plt.xlabel("time [s]"); plt.ylabel("thrust [N]"); plt.legend(); plt.title("Control Input")
        p3 = os.path.join(outdir, "plots", "thrust.png"); plt.savefig(p3, dpi=160); plt.close()

        # 4) One-step prediction vs actual (altitude)
        plt.figure()
        plt.plot(hist['t'][:-1], hist['h'][1:], label="h(k+1) actual")
        plt.plot(hist['t'][:-1], hist['pred_h'][:-1], label="h(k+1) predicted")
        plt.xlabel("time [s]"); plt.ylabel("altitude [m]"); plt.legend(); plt.title("One-step Prediction (Altitude)")
        p4 = os.path.join(outdir, "plots", "prediction_altitude.png"); plt.savefig(p4, dpi=160); plt.close()

        # 5) One-step prediction error histogram (altitude)
        import numpy as np
        pred_err = np.array(hist['h'][1:]) - np.array(hist['pred_h'][:-1])
        plt.figure()
        plt.hist(pred_err, bins=40)
        plt.xlabel("error [m]"); plt.ylabel("count"); plt.title("One-step Prediction Error (Altitude)")
        p5 = os.path.join(outdir, "plots", "prediction_error_hist_altitude.png"); plt.savefig(p5, dpi=160); plt.close()

    return outdir

def main():
    hist, mats = run_closed_loop(cfg)
    outdir = save_artifacts(cfg, hist, mats)
    return outdir

#if __name__ == "__main__":
#    outdir = main()
#    print("Artifacts saved to:", outdir)
#    outdir = main()
#    print("Artifacts saved to:", outdir)
#    print("Artifacts saved to:", outdir)
