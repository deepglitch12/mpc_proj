import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def dynamics(y,params,F):
    x, theta1, theta2, v, omega1, omega2 = y
    M, m1, m2 = params["M"], params["m1"], params["m2"]
    L1, L2 = params["L1"], params["L2"]
    l1, l2 = params["l1"], params["l2"]
    I1, I2 = params["I1"], params["I2"]
    g = params["g"]

    A = np.array([
        [M + m1 + m2, (m1 * l1 + m2 * L1) * np.cos(theta1), m2 * l2 * np.cos(theta2)],
        [(m1 * l1 + m2 * L1) * np.cos(theta1), m1 * (l1**2) + m2 * (L1**2) + I1, m2 * L1 * l2 * np.cos(theta1 - theta2)],
        [m2 * l2 * np.cos(theta2), m2 * L1 * l2 * np.cos(theta1 - theta2), m2 * (l2**2) + I2]
    ])

    B = np.array([
        F + (m1 * l1 + m2 * L1) * np.sin(theta1) * omega1**2 + m2 * l2 * np.sin(theta2) * omega2**2,
        (m1 * l1 + m2 * L1) * g * np.sin(theta1) - m2 * L1 * l2 * np.sin(theta1 - theta2) * omega2**2,
        m2 * g * l2 * np.sin(theta2) + m2 * L1 * l2 * np.sin(theta1 - theta2) * omega1**2
    ])

    # Solve for accelerations
    x_ddot, theta1_ddot, theta2_ddot = np.linalg.solve(A, B)

    #returning the derivative of the states
    return np.array([v, omega1, omega2, x_ddot, theta1_ddot, theta2_ddot])

def dist_noise_dynamics(y,params,F,d):

    x, theta1, theta2, v, omega1, omega2 = y
    M, m1, m2 = params["M"], params["m1"], params["m2"]
    L1, L2 = params["L1"], params["L2"]
    l1, l2 = params["l1"], params["l2"]
    I1, I2 = params["I1"], params["I2"]
    g = params["g"]

    # adding distrubance to the force
    f_net = F + d 

    A = np.array([
        [M + m1 + m2, (m1 * l1 + m2 * L1) * np.cos(theta1), m2 * l2 * np.cos(theta2)],
        [(m1 * l1 + m2 * L1) * np.cos(theta1), m1 * (l1**2) + m2 * (L1**2) + I1, m2 * L1 * l2 * np.cos(theta1 - theta2)],
        [m2 * l2 * np.cos(theta2), m2 * L1 * l2 * np.cos(theta1 - theta2), m2 * (l2**2) + I2]
    ])

    B = np.array([
        f_net + (m1 * l1 + m2 * L1) * np.sin(theta1) * omega1**2 + m2 * l2 * np.sin(theta2) * omega2**2,
        (m1 * l1 + m2 * L1) * g * np.sin(theta1) - m2 * L1 * l2 * np.sin(theta1 - theta2) * omega2**2,
        m2 * g * l2 * np.sin(theta2) + m2 * L1 * l2 * np.sin(theta1 - theta2) * omega1**2
    ])

    # Solve for accelerations
    x_ddot, theta1_ddot, theta2_ddot = np.linalg.solve(A, B)

    new_y = np.array([v, omega1, omega2, x_ddot, theta1_ddot, theta2_ddot]) 

    # print(new_y)

    #returning the derivative of the states
    return new_y


# RK4 Integration
def rk4_step(xk, dt, params,F):
    k1 = dt * dynamics(xk, params,F)
    k2 = dt * dynamics(xk + k1 / 2,params,F)
    k3 = dt * dynamics(xk + k2 / 2,params,F)
    k4 = dt * dynamics(xk + k3,params,F)
    return xk + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# RK4 Integration
def rk4_step_noise(xk, dt, params, F, d, Cd):
    k1 = dt * dist_noise_dynamics(xk, params,F,d)
    k2 = dt * dist_noise_dynamics(xk + k1 / 2,params,F,d)
    k3 = dt * dist_noise_dynamics(xk + k2 / 2,params,F,d)
    k4 = dt * dist_noise_dynamics(xk + k3,params,F,d)
    dist = (Cd*d).reshape(1,-1)

    return xk + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + dist[0] 





