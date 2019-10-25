import numpy as np

class Turtlebot():
    def __init__(self, x, r, phi, z_t):
        self.__dict__.update(locals())
        self.x_idx = 0
        self.z_idx = 0

    def step(self, t):
        x = self.x[:, self.x_idx:self.x_idx+1]
        self.x_idx += 1

        z = None
        while self.z_t[self.z_idx + 1] < t:
            self.z_idx += 1

        meas_update = True
        r = self.r[:, self.z_idx]
        phi = self.phi[:, self.z_idx]
        z = np.block([[r], [phi]])
        if self.z_idx < len(self.r):
            self.z_idx += 1
        if np.sum(np.isnan(r)) > 8:
            meas_update = False

        return x, z, meas_update
