#!/usr/bin/python3

import sys
import numpy as np
from scipy.io import loadmat
from visualizer import Visualizer
from turtlebot import Turtlebot
from particle_filter import ParticleFilter

__usage__ = '\nUsage:\tpython3 main.py <filename>.mat [True/False->animate]'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == "__main__":
    if sys.version_info[0] < 3:
        __error__("Python 3 is required" + __usage__)
    animate = True
    args = sys.argv[1:]
    if len(args) == 0:
        mat_file = 'processed_data.mat'
    elif len(args) < 3:
        mat_file = args[0]
        if not mat_file[-4:] == '.mat':
            __error__("Invalid file type" + __usage__)
        if len(args) == 2:
            animate = args[1][0] == 'T' or args[1][0] == 't'
    else:
        __error__("Too many arguments received" + __usage__)

    data = loadmat(mat_file)
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
            limits=lims, live=animate)

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
