import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from non_linear_dynamics import rk4_step, dynamics, rk4_step_noise, dist_noise_dynamics
import json
from IPython.display import HTML
from pathlib import Path

def animated(states,time,params,force,name):
    # Extract state variables
    x_vals = states[:, 0]   
    theta1_vals = states[:, 1]
    theta2_vals = states[:, 2]
    force_vals = force
    L1, L2 = params["L1"], params["L2"]
    l1, l2 = params["l1"], params["l2"]

    # Compute positions of pendulums
    cart_y = 0
    pendulum1_x = x_vals + L1 * np.sin(theta1_vals)
    pendulum1_y = cart_y + L1 * np.cos(theta1_vals)

    pendulum2_x = pendulum1_x + L2 * np.sin(theta2_vals)
    pendulum2_y = pendulum1_y + L2 * np.cos(theta2_vals)

    force_x = x_vals + force_vals
    force_y = cart_y

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Double Pendulum on a Cart")
    

    # Plot elements
    cart, = ax.plot([], [], 'ks', markersize=20)  # Cart
    rod1, = ax.plot([], [], 'b-', lw=2)  # First pendulum
    rod2, = ax.plot([], [], 'r-', lw=2)  # Second pendulum
    arrow, = ax.plot([],[], 'y-', lw=1)

    def init():
        cart.set_data([], [])
        rod1.set_data([], [])
        rod2.set_data([], [])
        arrow.set_data([],[])
        return cart, rod1, rod2, arrow

    def update(frame):
        cart.set_data([x_vals[frame]], [cart_y])
        rod1.set_data([x_vals[frame], pendulum1_x[frame]], [cart_y, pendulum1_y[frame]])
        rod2.set_data([pendulum1_x[frame], pendulum2_x[frame]], [pendulum1_y[frame], pendulum2_y[frame]])
        arrow.set_data([x_vals[frame],force_x[frame]],[cart_y, force_y])
        return cart, rod1, rod2

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, blit=False, interval=10)
    print()
    
    plt.show()
    ani.save(name, writer="pillow", fps=20)


if __name__ == "__main__":
    # Time setup
    dt = 0.01
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

    dist = [0]

    # Simulation loop
    for t in time:
        if t<T/2:
            y = rk4_step_noise(y, dt,params,0,0.1,noise_std=noise_std)
            dist.append(1)
        else:
            y = rk4_step_noise(y, dt,params,0,0,noise_std=noise_std)
            dist.append(0)
        states.append(y)

    # Convert to array for analysis
    states = np.array(states)

    dist = np.array(dist)

    animated(states, time, params, dist,"test_sim.gif")
