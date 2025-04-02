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

def dist_noise_dynamics(y,params,F,d,noise_std):

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

    #simulating the noise
    w = np.random.multivariate_normal(np.array([0,0,0,0,0,0]),noise_std)

    # print(w)

    new_y = np.array([v, omega1, omega2, x_ddot, theta1_ddot, theta2_ddot]) + w

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
def rk4_step_noise(xk, dt, params,F,d,noise_std):
    k1 = dt * dist_noise_dynamics(xk, params,F,d,noise_std)
    k2 = dt * dist_noise_dynamics(xk + k1 / 2,params,F,d,noise_std)
    k3 = dt * dist_noise_dynamics(xk + k2 / 2,params,F,d,noise_std)
    k4 = dt * dist_noise_dynamics(xk + k3,params,F,d,noise_std)
    return xk + (k1 + 2 * k2 + 2 * k3 + k4) / 6

if __name__ == "__main__":
    # Time setup
    dt = 0.001
    T = 10
    time = np.arange(0, T, dt)

    path_in_dir_script = Path(__file__).parent #fold where the main script is
    path_out_dir = path_in_dir_script / "../out/MPC"
    path_out_dir.mkdir(exist_ok=True)

    with open(path_in_dir_script/"config.json", "r") as file:
        params = json.load(file) 

    # Initial conditions
    y = np.array([0, 0.1, 0.1, 0, 0, 0])
    states = [y]

    noise_std = np.eye(y.shape[0])

    dist = []

    # Simulation loop
    for t in time:
        if t<T/2:
            y = rk4_step_noise(y, dt,params,0,1,noise_std=noise_std)
            dist.append([1])
        else:
            y = rk4_step_noise(y, dt,params,0,0,noise_std=noise_std)
            dist.append([0])
        states.append(y)

    # Convert to array for analysis
    states = np.array(states)

    dist = np.array(dist)





