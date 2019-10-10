import numpy as np
from numpy.random import randn as randn

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

def rand(size=(), min_=0, max_=1):
    return min_ + np.random.rand(*size)*(max_ - min_)

class MotionModel():
    def __init__(self, ts=0.1):
        self.dt = ts

    def __call__(self, u, x_m1):
        n0, = np.where(u[1] != 0)    # non-zero indices of omega
        vhat = u[0] 
        what = u[1] 
        temp = vhat[n0] / what[n0] 
        w_dt = what * self.dt
        theta = x_m1[2][n0]

        x = np.zeros(x_m1.shape)
        x[0][n0] = x_m1[0][n0] + temp*(np.sin(theta+w_dt[n0])-np.sin(theta))
        x[1][n0] = x_m1[1][n0] + temp*(np.cos(theta)-np.cos(theta+w_dt[n0]))
        x[2] = wrap(x_m1[2] + w_dt)

        if len(n0) != len(u[1]): 
            y0, = np.where(u[1] == 0) # zero indices of omega
            theta = x_m1[2][y0]
            x[0][y0] = x_m1[0][y0] + vhat[y0]*self.dt*np.cos(theta)
            x[1][y0] = x_m1[1][y0] + vhat[y0]*self.dt*np.sin(theta)
        return x

class MeasurementModel():
    def __init__(self):
        pass

    def __call__(self, states, mx, my):
        x_diff, y_diff = mx - states[0], my - states[1]
        r = np.sqrt(x_diff**2 + y_diff**2)
        phi = np.arctan2(y_diff, x_diff) - states[2]
        return np.block([[r], [phi]])
