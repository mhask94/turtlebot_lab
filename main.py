#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn
from scipy.io import loadmat
from visualizer import Visualizer
from utils import wrap, rand, MotionModel, MeasurementModel

class Turtlebot():
    def __init__(self, x, r, phi, z_t):
        self.__dict__.update(locals())
        self.x_idx = 0
        self.z_idx = 0

    def step(self, t):
        x = self.x[:,self.x_idx:self.x_idx+1]
        self.x_idx += 1

        meas_update = False
        z = None
        if t >= self.z_t[self.z_idx]:
            meas_update = True
            r = self.r[:,self.z_idx]
            phi = self.phi[:,self.z_idx]
            z = np.block([[r], [phi]])
            if self.z_idx < len(self.r):
                self.z_idx += 1

        return x, z, meas_update

class ParticleFilter():
    def __init__(self, alphas, Q, num_particles=1000, limits=[-10,10,-10,10],
            landmarks=np.empty(0)):
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.Q = Q.diagonal().reshape((len(Q),1))
        self.M = num_particles
        self.landmarks = landmarks
        self.g = MotionModel()
        self.h = MeasurementModel()

        self.chi = np.empty((4,num_particles))
        self.chi[0] = rand(self.chi[0].shape, limits[0], limits[1])
        self.chi[1] = rand(self.chi[1].shape, limits[2], limits[3])
        self.chi[2] = rand(self.chi[2].shape, -np.pi, np.pi)
        self.chi[3] = 1 / num_particles
        self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
        mu_diff = wrap(self.chi[:3] - self.mu, dim=2)
        self.sigma = np.cov(mu_diff)
        self.z_hat = np.ones((2,len(self.landmarks)))*50

    def _gauss_prob(self, diff, var):
        return np.exp(-diff**2/2/var) / np.sqrt(2*np.pi*var)

    def _low_var_resample(self):
        M_inv = 1/self.M
        r = rand(min_=0, max_=M_inv)
        c = self.chi[-1][0]
        i = 0
        for m in range(self.M):
            U = r + (m-1)*M_inv
            while U > c:
                i += 1
                c += self.chi[-1][i]
            self.chi[:3,m] = self.chi[:3,i]

    def predictionStep(self, u, dt): # u is np.array size 2x1
        # add noise to commanded inputs
        u_noisy = np.zeros((len(u), self.M))
        vsig = np.sqrt(self.a1*u[0]**2 + self.a2*u[1]**2)
        wsig = np.sqrt(self.a3*u[0]**2 + self.a4*u[1]**2)
        u_noisy[0] = u[0] + vsig*randn(self.M)
        u_noisy[1] = u[1] + wsig*randn(self.M)
        # propagate dynamics through motion model
        self.chi[:3] = self.g(u_noisy, self.chi[:3], dt)
        # update mu
        self.mu = np.mean(self.chi[:3], axis=1, keepdims=True)

    def correctionStep(self, z):
        self.chi[-1] = 1
        for i, (mx,my) in enumerate(self.landmarks):
            Zi = self.h(self.chi[:3], mx, my)
            if np.isnan(z[0,i]):
                continue
            diff = wrap(Zi - z[:,i:i+1], dim=1)
            z_prob = np.prod(self._gauss_prob(diff, 2*self.Q), axis=0)
            z_prob /= np.sum(z_prob)
            self.chi[-1] *= z_prob
            self.z_hat[:,i] = np.sum(z_prob * Zi, axis=1)

        self.chi[-1] /= np.sum(self.chi[-1])
        self._low_var_resample()

        self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
        mu_diff = wrap(self.chi[:3] - self.mu, dim=2)
        self.sigma = np.cov(mu_diff)

if __name__ == "__main__":
    data = loadmat('processed_data.mat')
    landmarks = data['landmarks']
    l_time = data['l_time'].flatten()
    l_depth = data['l_depth']
    l_bearing = data['l_bearing']
    odom_t = data['odom_t'].flatten()
    pos_odom_se2 = data['pos_odom_se2']
    vel_odom = data['vel_odom']
    del data

    alphas = np.array([0.1,0.01,0.01,0.1])
    Q = np.diag([0.1, 0.05])**2

    turtlebot = Turtlebot(pos_odom_se2, l_depth, l_bearing, l_time)
    
    lims = [-6, 12, -2, 16]
    M = 1000
    pf = ParticleFilter(alphas, Q, M, lims, landmarks)
    chi0 = pf.chi
    mu0 = pf.mu
    sigma0 = pf.sigma

    viz = Visualizer(odom_t[0], pos_odom_se2[:,0], chi0, mu0, sigma0, landmarks, 
            limits=lims, live=True)

    for i in range(len(odom_t)):
        if i == 0:
            continue # skip first step because visualizer already has data

        t = odom_t[i]
        dt = t - odom_t[i-1]
        u_c = vel_odom[:,i:i+1]

        x, z, got_meas = turtlebot.step(t)

        pf.predictionStep(u_c, dt)
        if got_meas:
            pf.correctionStep(z)

        viz.update(t, x, pf.chi, pf.mu, pf.sigma, pf.z_hat, got_meas)

viz.plotHistory()
