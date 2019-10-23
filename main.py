#!/usr/bin/python3

import sys
import numpy as np
from scipy.io import loadmat
from visualizer import Visualizer
from turtlebot import Turtlebot
from particle_filter import ParticleFilter
from utils import wrap

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
        mat_file_truth = 'truth_data.mat'
    elif len(args) < 3:
        mat_file = args[0]
        if not mat_file[-4:] == '.mat':
            __error__("Invalid file type" + __usage__)
        if not mat_file_truth[-4:] == '.mat':
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

    truth_data = loadmat(mat_file_truth)
    t_truth = truth_data['t_truth'].flatten()
    th_truth = truth_data['th_truth']
    x_truth = truth_data['x_truth']
    y_truth = truth_data['y_truth']

    # Cut down truth data to align with odom time
    i = 0
    j = 0
    while i < len(odom_t):
        if t_truth[i] < odom_t[j]:
            t_truth = np.delete(t_truth, i)
            th_truth = np.delete(th_truth, i)
            x_truth = np.delete(x_truth, i)
            y_truth = np.delete(y_truth, i)
        else:
            j += 1
            i += 1
    if len(odom_t) < len(t_truth):  # remove extra truth indices for plotting
        t_truth = np.delete(t_truth, -1)
        th_truth = np.delete(th_truth, -1)
        x_truth = np.delete(x_truth, -1)
        y_truth = np.delete(y_truth, -1)

    pos_truth_se2 = np.vstack((x_truth.flatten(), y_truth.flatten(), wrap(th_truth.flatten())))

    del data, truth_data

    alphas = np.array([0.1, 0.01, 0.01, 0.1])*10
    Q = np.diag([0.1, 0.05])**2

    turtlebot = Turtlebot(pos_truth_se2, l_depth, l_bearing, l_time)
    
    lims = [-5, 5, -5, 5]
    M = 1000
    pf = ParticleFilter(alphas, Q, M, lims, landmarks)
    chi0 = pf.chi
    mu0 = pf.mu
    sigma0 = pf.sigma

    viz = Visualizer(odom_t[0], pos_truth_se2[:, 0], chi0, mu0, sigma0, landmarks,
            limits=lims, live=animate)

    for i in range(len(odom_t)):
        if i == 0:
            continue  # skip first step because visualizer already has data

        t = odom_t[i]
        dt = t - odom_t[i-1]
        u_c = vel_odom[:, i:i+1]

        x, z, got_meas = turtlebot.step(t)

        pf.predictionStep(u_c, dt)

        # skip measurement updates when no commands given to wheels to avoid false convergence
        if all(i == 0 for i in u_c):
            continue
        else:
            if got_meas:
                pf.correctionStep(z)

        viz.update(t, x, pf.chi, pf.mu, pf.sigma, pf.z, got_meas)

viz.plotHistory()
