import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import dynamics,rk4_step
from draw import animated
from mpc_solver import mpc_solve
import math
from pathlib import Path


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/MPC"
path_out_dir.mkdir(exist_ok=True)

with open(path_in_dir_script/"config.json", "r") as file:
    params = json.load(file) 

#defining the constants
M, m1, m2 = params["M"], params["m1"], params["m2"]
L1, L2 = params["L1"], params["L2"]
l1, l2 = params["l1"], params["l2"]
I1, I2 = params["I1"], params["I2"]
g = params["g"]

#defining the matrices for the linearized system

J = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,M+m1+m2,(m1*l1+m2*L1),m2*l2],
              [0,0,0,(m1*l1+m2*L1),m1*(l1**2)+m2*(L1**2)+I1,m2*L1*l2],
              [0,0,0,m2*l2,m2*L1*l2,m2*(l2**2)+I2]])

N = np.array([[0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1],
              [0,0,0,0,0,0],
              [0,(m1*l1 + m2*L1)*g,0,0,0,0],
              [0,0,m2*l2*g,0,0,0]])

F = np.array([[0],
              [0],
              [0],
              [1],
              [0],
              [0]])


#continuous dynamic matrices

Ac = np.linalg.inv(J) @ N
Bc = np.linalg.inv(J) @ F

Cc = np.eye(Ac.shape[0])
Dc = np.zeros(Bc.shape)

#discretizing the system
dt= 0.01

dsys = sp.signal.cont2discrete((Ac,Bc,Cc,Dc),dt,method='zoh')

A = dsys[0]
B = dsys[1]
C = dsys[2]
D = dsys[3]

print(A)

print(B)
# LQR Control
#defining the weights

print(f"Rank of Controllability Matrix:",np.linalg.matrix_rank(ct.ctrb(A,B)))


R_values = [1, 10, 100]
N = 40

# Create figures and axes
fig_u, ax_u = plt.subplots()
fig_x, ax_x = plt.subplots()
fig_th1, ax_th1 = plt.subplots()
fig_th2, ax_th2 = plt.subplots()
fig_xdot, ax_xdot = plt.subplots()
fig_thdot, ax_thdot = plt.subplots()

for r_val in R_values:
    print(f"Running for: R = {r_val}")

    Q = sp.linalg.block_diag(1000,100,100,1000,0,0)

    R = r_val

    T = 10
    time = np.arange(0, T, dt)

    x_ref = np.array([0, math.radians(0), -math.radians(0), 0, 0, 0]).reshape(-1, 1)
    u_ref = 0
    theta_const = 0.174
    x_const = 3
    u_const = 100
    o = 10

    x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1, 1)
    x_lb = np.array([-x_const, -theta_const, -theta_const, -o, -o, -o]).reshape(-1, 1)
    u_ub = np.array(u_const).reshape(-1, 1)
    u_lb = np.array(-u_const).reshape(-1, 1)


    P, _, K = ct.dare(A, B, Q, R)

    y = np.array([0, -0.125, 0.125, 0, 0, 0])
    states = [y]
    control_inputs = [0]

    flag = True
    for t in time:
        x_cur = states[-1].reshape(-1, 1)
        u = mpc_solve(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref)
        if u is None:
            print(f"MPC failed for Q={r_val}")
            flag = False
            break

        print(f"Percentage done: {(t/T)*100:.1f}%")
        y = rk4_step(y, dt, params, u[0][0][0])
        control_inputs.append(u[0][0][0])
        states.append(y)

    if flag:
        states = np.array(states)

        label = f"R={r_val}"

        ax_u.step(time, control_inputs[:-1], label=label)
        ax_x.step(time, states[1:, 0], label=label)
        ax_th1.step(time, np.rad2deg(states[1:, 1]), label=f"Theta1 {label}")
        ax_th2.step(time, np.rad2deg(states[1:, 2]), label=f"Theta2 {label}")
        ax_xdot.step(time, states[1:, 3], label=label)
        ax_thdot.step(time, np.rad2deg(states[1:, 4]), label=f"Theta1_dot {label}")
        ax_thdot.step(time, np.rad2deg(states[1:, 5]), label=f"Theta2_dot {label}")

# Finalize and save plots
for ax, title, xlabel, ylabel, fname in [
    (ax_u, "Control Input", "Time [s]", "u [N]", "Control_input_R.png"),
    (ax_x, "Position x", "Time [s]", "x [m]", "Position_R.png"),
    (ax_th1, "Angles Theta1", "Time [s]", "Angle [deg]", "theta1_R.png"),
    (ax_th2, "Angles Theta2", "Time [s]", "Angle [deg]", "theta2_R.png"),    
    (ax_xdot, "Velocity x_dot", "Time [s]", "xdot [m/s]", "Position_dot_R.png"),
    (ax_thdot, "Angular Velocities", "Time [s]", "deg/s", "theta_dot_R.png")
]:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig = ax.get_figure()
    fig.savefig(path_out_dir / fname)
