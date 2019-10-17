import numpy as np
from numpy.random import randn as randn

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle


class MotionModel():
    def __init__(self, alphas, noise=True):
        self.a1, self.a2, self.a3, self.a4 = alphas[:4]
        if len(alphas) == 6:
            self.a5, self.a6 = alphas[4:6]
        else:
            self.a5, self.a6 = 0, 0
        self.noise = noise

    def __call__(self, u, x_m1, dt):
        # add noise if needed
        u_noisy = np.zeros((len(u), len(x_m1[0])))
        vsig = np.sqrt(self.a1*u[0]**2 + self.a2*u[1]**2) * self.noise
        wsig = np.sqrt(self.a3*u[0]**2 + self.a4*u[1]**2) * self.noise
        gamsig = np.sqrt(self.a5*u[0]**2 + self.a6*u[1]**2) * self.noise
        u_noisy[0] = u[0] + vsig*randn(len(x_m1[0]))
        u_noisy[1] = u[1] + wsig*randn(len(x_m1[1]))

        n0, = np.where(u_noisy[1] != 0)    # non-zero indices of omega
        vhat = u_noisy[0] 
        what = u_noisy[1] 
        temp = vhat[n0] / what[n0] 
        w_dt = what * dt
        gamma_dt = gamsig*randn(len(w_dt)) * dt
        theta = x_m1[2][n0]

        x = np.zeros(x_m1.shape)
        x[0][n0] = x_m1[0][n0] + temp*(np.sin(theta+w_dt[n0])-np.sin(theta))
        x[1][n0] = x_m1[1][n0] + temp*(np.cos(theta)-np.cos(theta+w_dt[n0]))
        x[2] = wrap(x_m1[2] + w_dt + gamma_dt)

        if len(n0) != len(u_noisy[1]): 
            y0, = np.where(u_noisy[1] == 0) # zero indices of omega
            theta = x_m1[2][y0]
            x[0][y0] = x_m1[0][y0] + vhat[y0]*dt*np.cos(theta)
            x[1][y0] = x_m1[1][y0] + vhat[y0]*dt*np.sin(theta)
        return x


class MeasurementModel():
    def __init__(self):
        pass

    def __call__(self, states, mx, my):
        x_diff, y_diff = mx - states[0], my - states[1]
        r = np.sqrt(x_diff**2 + y_diff**2)
        phi = np.arctan2(y_diff, x_diff) - states[2]
        return np.block([[r], [wrap(phi)]])

