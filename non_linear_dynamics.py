import numpy as np
import json
import matplotlib.pyplot as plt

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

# RK4 Integration
def rk4_step(xk, dt, params,F):
    k1 = dt * dynamics(xk, params,F)
    k2 = dt * dynamics(xk + k1 / 2,params,F)
    k3 = dt * dynamics(xk + k2 / 2,params,F)
    k4 = dt * dynamics(xk + k3,params,F)
    return xk + (k1 + 2 * k2 + 2 * k3 + k4) / 6

if __name__ == "__main__":
    # Time setup
    dt = 0.001
    T = 10
    time = np.arange(0, T, dt)

    with open("config.json", "r") as file:
        params = json.load(file) 

    # Initial conditions
    y = np.array([0, 0.1, 0.1, 0, 0, 0])
    states = [y]

    # Simulation loop
    for t in time:
        if t==0:
            y = rk4_step(y, dt,params,0)
        else:
            y = rk4_step(y, dt,params,0)
        states.append(y)

    # Convert to array for analysis
    states = np.array(states)

    states[:,1] = states[:,1]%(2 * np.pi) - np.pi
    states[:,2] = states[:,2]%(2 * np.pi) - np.pi
    plt.plot(states[:,2])
    plt.savefig("theta1.png")
    plt.figure()
    plt.plot(states[:,4])
    plt.savefig("theta2.png")


