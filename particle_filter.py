import numpy as np
from numpy.random import randn
from utils import wrap, MotionModel, MeasurementModel

class ParticleFilter():
    def __init__(self, alphas, Q, num_particles=1000, limits=[-5, 5, -5, 5],
        landmarks=np.empty(0)):
        self.Q = Q.diagonal().reshape((len(Q), 1))
        self.M = num_particles
        self.landmarks = landmarks
        self.g = MotionModel(alphas, noise=True)
        self.h = MeasurementModel()

        self.chi = np.empty((4, num_particles))
        self.chi[0] = np.random.uniform(limits[0], limits[1], self.chi[0].shape)
        self.chi[1] = np.random.uniform(limits[2], limits[3], self.chi[1].shape)
        self.chi[2] = np.random.uniform(-np.pi, np.pi, self.chi[2].shape)
        self.chi[3] = 1 / num_particles
        self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
        mu_diff = wrap(self.chi[:3] - self.mu, dim=2)
        self.sigma = np.cov(mu_diff)
        self.z_nan = 50
        self.z_hat = np.ones((2, len(self.landmarks)))*self.z_nan
        self.z = np.ones((2, len(self.landmarks)))*self.z_nan
        self.n = 3

    def _gauss_prob(self, diff, var):
        return np.exp(-diff**2/2/var) / np.sqrt(2*np.pi*var)

    def _low_var_resample(self):
        M_inv = 1/self.M
        r = np.random.uniform(low=0, high=M_inv)
        c = np.cumsum(self.chi[-1])
        U = np.arange(self.M)*M_inv + r
        diff = c - U[:,None]
        i = np.argmax(diff > 0, axis=1)

        n = 3 # num states

        P = np.cov(self.chi[:n])
        self.chi = self.chi[:,i]

        uniq = np.unique(i).size
        if uniq*M_inv < 0.1:
            Q = P / ((self.M*uniq)**(1/n))
            noise = Q @ randn(*self.chi[:n].shape)
            self.chi[:n] = wrap(self.chi[:n] + noise, dim=2)
        self.chi[-1] = M_inv
        
    def predictionStep(self, u, dt):  # u is np.array size 2x1
        # propagate dynamics through motion model
        self.chi[:3] = self.g(u, self.chi[:3], dt)
        # update mu
        self.mu = np.mean(self.chi[:3], axis=1, keepdims=True)
        self.mu[2] = wrap(self.mu[2])

    def correctionStep(self, z):
        self.chi[-1] = 1
        self.z = np.ones((2, len(self.landmarks))) * self.z_nan
        for i, (mx, my) in enumerate(self.landmarks):
            Zi = self.h(self.chi[:3], mx, my)
            if np.isnan(z[0, i]):
                continue
            diff = wrap(Zi - z[:, i:i+1], dim=1)
            z_prob = np.prod(self._gauss_prob(diff, 2*self.Q), axis=0)
            z_prob /= np.sum(z_prob)
            self.chi[-1] *= z_prob
            self.z_hat[:, i] = np.sum(z_prob * Zi, axis=1)
            self.z[:, i] = z[:, i]

        self.chi[-1] /= np.sum(self.chi[-1])
        self._low_var_resample()

        self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
        mu_diff = wrap(self.chi[:3] - self.mu, dim=2)
        self.sigma = np.cov(mu_diff)
