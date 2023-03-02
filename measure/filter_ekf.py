#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv


class Ekf():
    
    def __init__(self, x_0_init):
        self.xcnt   = 0
        self.x_esti = None
        self.P      = None

        self.dt = 0.05
        self.A = np.eye(3) + self.dt * np.array([[0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        self.H = np.zeros((1, 3))
        self.Q = np.array([[0, 0, 0],
                    [0, 0.001, 0],
                    [0, 0, 0.001]])
        self.R = np.array([[10]])
        self.x_0 = np.array([int(x_0_init), 0, 0])
        self.P_0 = 10 * np.eye(3)

    def Ajacob_at(self, x_esti):
        return self.A
    
    def Hjacob_at(self, x_pred):
        self.H[0][0] = x_pred[0] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
        self.H[0][1] = 0
        self.H[0][2] = x_pred[2] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
        return self.H
        
    def fx(self, x_esti):
        return self.A @ x_esti
    
    def hx(self, x_pred):
        z_pred = np.sqrt(x_pred[0]**2 + x_pred[2]**2)
        return np.array([z_pred])
    
    def extended_kalman_filter(self, z_meas, x_esti, P):
        self.A = self.Ajacob_at(x_esti)
        x_pred = self.fx(x_esti)
        P_pred = self.A @ P @ self.A.T + self.Q
    
        self.H = self.Hjacob_at(x_pred)
        K = P_pred @ self.H.T @ inv(self.H @ P_pred @ self.H.T + self.R)

        x_esti = x_pred + K @ (z_meas - self.hx(x_pred))
    
        P = P_pred - K @ self.H @ P_pred
        return x_esti, P

    def start(self, x):
        z_meas = np.array([x])
        if self.xcnt == 0:
            self.x_esti, self.P = self.x_0, self.P_0
            self.xcnt += 1
        else:
            self.x_esti, self.P = self.extended_kalman_filter(z_meas, self.x_esti, self.P)
    
        return int(self.x_esti[0])#self.x_esti  #int(self.x_esti[0])
    
    
def get_test_dataset():
    # start = 0
    # end = 2 * np.pi
    # dx = 1000
    # x = np.linspace(start, end, dx)
    # return x, np.sin(x)
    SIG_AMPLITUDE = 10
    SIG_OFFSET = 2
    SIG_PERIOD = 200
    NOISE_AMPLITUDE = 3
    N_SAMPLES = 1 * SIG_PERIOD
    INSTRUMENT_RANGE = 9

    # construct a sine wave
    times = np.arange(N_SAMPLES).astype(float)
    signal = SIG_AMPLITUDE * np.sin(2 * np.pi * times / SIG_PERIOD) + SIG_OFFSET

    # and mix it with some random noise
    noise = NOISE_AMPLITUDE * np.random.normal(size=N_SAMPLES)
    signal += noise
    return times, signal


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib

    time, raw = get_test_dataset()
    kalman = Ekf(0)

    fig = plt.figure(figsize=(8, 8))
    try:
        for i in range(len(time)):
        
            pass
            # plt.scatter(time[i], raw[i], c="r", marker='o')
            # esti = kalman.start(time[i], raw[i])
            # plt.scatter(esti[0], esti[2], c="y", marker='x')

    except KeyboardInterrupt:
        print('Ctrl + C')    
        
    plt.show()