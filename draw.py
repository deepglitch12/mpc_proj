import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from non_linear_dynamics import rk4_step, dynamics
import json


def animates(states,time,params):
    # Extract state variables
    x_vals = states[:, 0]   
    theta1_vals = states[:, 2]
    theta2_vals = states[:, 4]

    L1, L2 = params["L1"], params["L2"]
    l1, l2 = params["l1"], params["l2"]

    # Compute positions of pendulums
    cart_y = 0
    pendulum1_x = x_vals + L1 * np.sin(theta1_vals)
    pendulum1_y = cart_y + L1 * np.cos(theta1_vals)

    pendulum2_x = pendulum1_x + L2 * np.sin(theta2_vals)
    pendulum2_y = pendulum1_y + L2 * np.cos(theta2_vals)

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Double Pendulum on a Cart")

    # Plot elements
    cart, = ax.plot([], [], 'ks', markersize=20)  # Cart
    rod1, = ax.plot([], [], 'b-', lw=2)  # First pendulum
    rod2, = ax.plot([], [], 'r-', lw=2)  # Second pendulum

    def init():
        cart.set_data([], [])
        rod1.set_data([], [])
        rod2.set_data([], [])
        return cart, rod1, rod2

    def update(frame):
        cart.set_data([x_vals[frame]], [cart_y])
        rod1.set_data([x_vals[frame], pendulum1_x[frame]], [cart_y, pendulum1_y[frame]])
        rod2.set_data([pendulum1_x[frame], pendulum2_x[frame]], [pendulum1_y[frame], pendulum2_y[frame]])
        return cart, rod1, rod2

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, blit=False, interval=10)
    plt.show()

if __name__ == '__main__':
    
    with open("config.json", "r") as file:
        params = json.load(file)
    # Define parameters
    dt = 0.01
    T = 10
    time = np.arange(0, T, dt)

    # Initial conditions
    y = np.array([0, 0, 0, 0, 0, 0])
    states = [y]

    # Simulation loop
    for t in time:
        if t % 2 == 0:
            y = rk4_step(y, dt,params,0)
        else:
            y = rk4_step(y, dt,params,0)
        states.append(y)

    states = np.array(states)

    animates(states,time,params)

