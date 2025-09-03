
#Hunter Weston
#Vertical rocket dynamics with fuel consumption & q drag
#State x = [h, v, m] (altitude, vertical velocity, mass)
#Control u = thrust [N]
#Dynamics:
    #h_dot = v
    #v_dot = (u - D(v) - m*g0)/m + w(t)  (where D(v)=0.5*rho*v*|v|*Cd*A)
    #m_dot = -u/(Isp * g0)

import numpy as np

class Rocket1D:
    def __init__(self, phys):
        self.p = phys

    def drag(self, v):
        return 0.5 * self.p.rho * v * abs(v) * self.p.Cd * self.p.A

    def derivatives(self, x, u, wind_accel=0.0):
        h, v, m = x
        u = float(u)
        m = max(m, self.p.m_dry)
        D = self.drag(v)
        hdot = v
        vdot = (u - D - m*self.p.g0)/m + wind_accel
        mdot = -u/(self.p.Isp*self.p.g0)
        if m <= self.p.m_dry and mdot < 0.0:
            mdot = 0.0
        return np.array([hdot, vdot, mdot], dtype=float)

    def step(self, x, u, dt, wind_accel=0.0):
        #RK4
        k1 = self.derivatives(x, u, wind_accel)
        k2 = self.derivatives(x + 0.5*dt*k1, u, wind_accel)
        k3 = self.derivatives(x + 0.5*dt*k2, u, wind_accel)
        k4 = self.derivatives(x + dt*k3, u, wind_accel)
        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_next[2] = max(x_next[2], self.p.m_dry)
        return x_next

def rocket_altitude_parabola(total_time, apogee=80.0):
    t_peak = total_time / 2
    coeffs = np.polyfit([0, t_peak, total_time], [0, apogee, 0], 2)

    def s(t):
        return max(np.polyval(coeffs, t), 0)

    def v(t):
        return np.polyval(np.polyder(coeffs), t)

    return s, v, coeffs