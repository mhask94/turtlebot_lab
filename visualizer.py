#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils import wrap

class Visualizer:
    def __init__(self, t0, x0, particles, xhat0, sigma0, landmarks=np.empty(0),
            limits=[-5, 5, -5, 5], live=True):
        self.time_hist = [t0]

        x, y, theta = x0.flatten()
        self.x_hist = [x]
        self.y_hist = [y]
        self.theta_hist = [theta]
        
        self.xhat_hist = [xhat0.item(0)]
        self.yhat_hist = [xhat0.item(1)]
        self.thetahat_hist = [xhat0.item(2)]

        self.xerr_hist = [x - xhat0.item(0)]
        self.yerr_hist = [y - xhat0.item(1)]
        self.thetaerr_hist = [wrap(theta - xhat0.item(2))]

        self.x2sig_hist = [2*np.sqrt(sigma0[0, 0].item())]
        self.y2sig_hist = [2*np.sqrt(sigma0[1, 1].item())]
        self.theta2sig_hist = [2*np.sqrt(sigma0[2, 2].item())]

        self.live = live
        if self.live:
            plt.rcParams["figure.figsize"] = (9, 7)
            self.fig, self.ax = plt.subplots()
            self.ax.axis(limits)
            self.ax.set_title('Turtlebot Simulation')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.R = 0.2
            self.circ = Circle((x, y), radius=self.R, color='y', ec='k')
            self.ax.add_patch(self.circ)
            xdata = [x, x + self.R*np.cos(theta)]
            ydata = [y, y + self.R*np.sin(theta)]
            self.line, = self.ax.plot(xdata, ydata, 'k')
            self.ax.plot(landmarks[:, 0], landmarks[:, 1], 'kx', label='landmark')

            self.true_dots, = self.ax.plot(self.x_hist,self.y_hist, 'b.',
                    markersize=1, label='truth')
            self.est_dots, = self.ax.plot(self.xhat_hist,self.yhat_hist, 'r.',
                    markersize=1, label='estimates')

            self.est_lms, = self.ax.plot(20, 20, 'rx', label='est landmark')
            self.particle_dots, = self.ax.plot(particles[0], particles[1], 'g.',
                    markersize=2, label='particles')

        self.ax.legend(loc='upper left')
        self._display()

    def update(self, t, true_pose, particles, est_pose, covariance, z, gotz):
        self.time_hist.append(t)

        x, y, theta = true_pose.flatten()
        self.x_hist.append(x)
        self.y_hist.append(y)
        self.theta_hist.append(theta)

        self.xhat_hist.append(est_pose.item(0))
        self.yhat_hist.append(est_pose.item(1))
        self.thetahat_hist.append(est_pose.item(2))

        self.xerr_hist.append(x - est_pose.item(0))
        self.yerr_hist.append(y - est_pose.item(1))
        self.thetaerr_hist.append(wrap(theta - est_pose.item(2)))

        self.x2sig_hist.append(2*np.sqrt(covariance[0, 0].item()))
        self.y2sig_hist.append(2*np.sqrt(covariance[1, 1].item()))
        self.theta2sig_hist.append(2*np.sqrt(covariance[2, 2].item()))

        if self.live:
            self.circ.set_center((x, y))
            self.line.set_xdata([x, x + self.R*np.cos(theta)])
            self.line.set_ydata([y, y + self.R*np.sin(theta)])

            self.true_dots.set_xdata(self.x_hist)
            self.true_dots.set_ydata(self.y_hist)
            self.est_dots.set_xdata(self.xhat_hist)
            self.est_dots.set_ydata(self.yhat_hist)

            self.particle_dots.set_xdata(particles[0])
            self.particle_dots.set_ydata(particles[1])

            if gotz:
                est_lms = np.zeros(z.shape)
                for i, (r, phi) in enumerate(z.T):
                    xi = est_pose.item(0) + r*np.cos(phi+est_pose.item(2))
                    yi = est_pose.item(1) + r*np.sin(phi+est_pose.item(2))
                    est_lms[:, i] = np.array([xi, yi])
                self.est_lms.set_xdata(est_lms[0, :])
                self.est_lms.set_ydata(est_lms[1, :])
        
        self._display()

    def plotHistory(self):
        if not self.live:
            self.true_dots, = self.ax.plot(self.x_hist, self.y_hist, 'b.',
                    markersize=3, label='truth')
            self.est_dots, = self.ax.plot(self.xhat_hist, self.yhat_hist, 'r.',
                    markersize=3, label='estimates')

        plt.rcParams["figure.figsize"] = (8,8)
        fig2, axes2 = plt.subplots(3, 1, sharex=True)
        axes2[0].plot(self.time_hist, self.x_hist, 'b', label='truth')
        axes2[0].plot(self.time_hist, self.xhat_hist, 'r', label='est')
        axes2[0].set_ylabel('X (m)')
        axes2[0].set_title('States vs Estimates')
        axes2[0].legend()

        axes2[1].plot(self.time_hist, self.y_hist, 'b', label='truth')
        axes2[1].plot(self.time_hist, self.yhat_hist, 'r', label='est')
        axes2[1].set_ylabel('Y (m)')
        axes2[1].legend()

        axes2[2].plot(self.time_hist, self.theta_hist, 'b', label='truth')
        axes2[2].plot(self.time_hist, self.thetahat_hist, 'r', label='est')
        axes2[2].set_ylabel('Theta (rad)')
        axes2[2].set_xlabel('Time (s)')
        axes2[2].legend()

        self.x2sig_hist = np.array(self.x2sig_hist)
        self.y2sig_hist = np.array(self.y2sig_hist)
        self.theta2sig_hist = np.array(self.theta2sig_hist)
        fig3, axes3 = plt.subplots(3, 1, sharex=True)
        axes3[0].plot(self.time_hist, self.xerr_hist, 'b', label='error')
        axes3[0].plot(self.time_hist, self.x2sig_hist, 'r', label='2$\sigma$')
        axes3[0].plot(self.time_hist, -self.x2sig_hist, 'r')
        axes3[0].set_ylabel('X (m)')
        axes3[0].set_title('Error Plots')
        axes3[0].legend()

        axes3[1].plot(self.time_hist, self.yerr_hist, 'b', label='error')
        axes3[1].plot(self.time_hist, self.y2sig_hist, 'r', label='2$\sigma$')
        axes3[1].plot(self.time_hist, -self.y2sig_hist, 'r')
        axes3[1].set_ylabel('Y (m)')
        axes3[1].legend()

        axes3[2].plot(self.time_hist, self.thetaerr_hist, 'b', label='error')
        axes3[2].plot(self.time_hist, self.theta2sig_hist, 'r', label='2$\sigma$')
        axes3[2].plot(self.time_hist, -self.theta2sig_hist, 'r')
        axes3[2].set_ylabel('Theta Error (rad)')
        axes3[2].set_xlabel('Time (s)')
        axes3[2].legend()

        plt.show()

    def _display(self):
        plt.pause(0.000001)
