
import numpy as np


class Rocket3D:
    def __init__(self, phys):
        self.p = phys

    def derivatives(self, x, u, wind_vx=0.0, wind_vy=0.0):
        """
        Full 3D rocket EOM.
        x = [h, vz, xp, vx, yp, vy, pitch, prate, yaw, yrate, m]
        u = [F, delta_p, delta_y]
        wind_vx, wind_vy: horizontal wind VELOCITY [m/s] (North, East)
        """
        h, vz, xp, vx, yp, vy, pitch, prate, yaw, yrate, m = x
        F  = float(np.clip(u[0], 0.0, 500.0))
        dp = float(np.clip(u[1], -self.p.delta_max, self.p.delta_max))
        dy = float(np.clip(u[2], -self.p.delta_max, self.p.delta_max))
        m  = max(m, self.p.m_dry)

        F_h  = F * np.cos(pitch + dp) * np.cos(yaw + dy)
        F_xp = -F * np.sin(pitch + dp)
        F_yp = -F * np.sin(yaw + dy)

        Dz = 0.5 * self.p.rho * vz * abs(vz) * self.p.Cd * self.p.A_ref
        vrel_x = vx - wind_vx
        vrel_y = vy - wind_vy
        Dx = 0.5 * self.p.rho * vrel_x * abs(vrel_x) * self.p.Cd_lat * self.p.A_lat
        Dy = 0.5 * self.p.rho * vrel_y * abs(vrel_y) * self.p.Cd_lat * self.p.A_lat

        q_axial = 0.5 * self.p.rho * max(abs(vz), 0.5)**2
        M_p_static = q_axial * self.p.A_ref * self.p.L_cp * np.sin(pitch)
        M_y_static = q_axial * self.p.A_ref * self.p.L_cp * np.sin(yaw)

        M_p_wind = -Dx * self.p.L_cp
        M_y_wind = -Dy * self.p.L_cp

        M_p_tvc = F * np.sin(dp) * self.p.L_tvc
        M_y_tvc = F * np.sin(dy) * self.p.L_tvc

        M_pitch = M_p_tvc + M_p_static + M_p_wind
        M_yaw   = M_y_tvc + M_y_static + M_y_wind

        mdot = -F / (self.p.Isp * self.p.g0)
        if m <= self.p.m_dry and mdot < 0.0:
            mdot = 0.0
            F_h = 0.0; F_xp = 0.0; F_yp = 0.0

        return np.array([
            vz,
            (F_h  - Dz) / m - self.p.g0,
            vx,
            (F_xp - Dx) / m,
            vy,
            (F_yp - Dy) / m,
            prate,
            M_pitch / self.p.Izz,
            yrate,
            M_yaw   / self.p.Izz,
            mdot,
        ], dtype=float)

    def step(self, x, u, dt, wind_vx=0.0, wind_vy=0.0):
        """RK4 integration step"""
        k1 = self.derivatives(x, u, wind_vx, wind_vy)
        k2 = self.derivatives(x + 0.5*dt*k1, u, wind_vx, wind_vy)
        k3 = self.derivatives(x + 0.5*dt*k2, u, wind_vx, wind_vy)
        k4 = self.derivatives(x +     dt*k3, u, wind_vx, wind_vy)
        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_next[10] = max(x_next[10], self.p.m_dry)   #mass floor
        return x_next
def rocket_ascent_reference(T_ascent, apogee=80.0):
    """Smooth raised-cosine reference for ascent. Returns (s_ref, vz_ref) callables."""
    T = max(float(T_ascent), 1e-6)
    def s(t):
        tc = np.clip(float(t) / T, 0.0, 1.0)
        return float(apogee * 0.5 * (1.0 - np.cos(np.pi * tc)))
    def vz(t):
        tc = np.clip(float(t) / T, 0.0, 1.0)
        return float(apogee * np.pi / (2.0 * T) * np.sin(np.pi * tc))
    return s, vz


def rocket_altitude_parabola(total_time, apogee=80.0):
    """Parabolic altitude profile over full flight: h(0)=0, h(T/2)=apogee, h(T)=0."""
    T = max(float(total_time), 1e-6)

    def s(t):
        tc = np.clip(float(t) / T, 0.0, 1.0)
        return float(4.0 * apogee * tc * (1.0 - tc))

    def vz_fn(t):
        tc = np.clip(float(t) / T, 0.0, 1.0)
        return float((4.0 * apogee / T) * (1.0 - 2.0 * tc))

    return s, vz_fn, np.zeros(11)


def build_realistic_3d_trajectory(t, h, v, u, thrust_max=30.0, lateral_scale=12.0):
    """Generate a smooth 3D path consistent with 1D ascent profiles for plotting/tests."""
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)

    n = min(len(t), len(h), len(v), len(u))
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    t = t[:n]
    h = h[:n]
    v = v[:n]
    u = u[:n]

    if n > 1:
        dt = np.diff(t)
        dt = np.clip(dt, 1e-6, None)
        dt = np.append(dt, dt[-1])
    else:
        dt = np.array([0.02], dtype=float)

    u_norm = np.clip(u / max(float(thrust_max), 1e-6), 0.0, 1.5)
    v_norm = np.tanh(v / max(np.max(np.abs(v)) + 1e-6, 1.0))

    a_x = lateral_scale * (0.08 * np.sin(0.7 * t) + 0.10 * u_norm - 0.04 * v_norm)
    a_y = lateral_scale * (0.07 * np.cos(0.55 * t + 0.4) + 0.06 * u_norm + 0.03 * v_norm)

    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)

    for i in range(1, n):
        vx[i] = vx[i - 1] + a_x[i - 1] * dt[i - 1]
        vy[i] = vy[i - 1] + a_y[i - 1] * dt[i - 1]
        x[i] = x[i - 1] + vx[i] * dt[i - 1]
        y[i] = y[i - 1] + vy[i] * dt[i - 1]

    x -= x[0]
    y -= y[0]
    taper = np.clip(h / max(np.max(h), 1e-6), 0.0, 1.0)
    x *= 0.3 + 0.7 * taper
    y *= 0.3 + 0.7 * taper

    z = h.copy()
    return x, y, z