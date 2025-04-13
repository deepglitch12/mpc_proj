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

Q = sp.linalg.block_diag(1000,100,100,0,0,0)

R = 100

#solution to dare
P ,_,K = ct.dare(A, B,Q,R)

T = 10
time = np.arange(0, T, dt)

y = np.array([0, -0.125, 0.125, 0, 0, 0])
# y = np.array([1, 0, 0, 0, 0, 0])
states = [y]
# print(y.shape)

#constraints
x_ref = np.array([0,math.radians(0),-math.radians(0),0,0,0]).reshape(-1,1)
u_ref = 0

theta_const = 0.174
x_const = 3
u_const = 100

x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
u_ub = np.array(u_const).reshape(-1,1)
u_lb = np.array(-u_const).reshape(-1,1)

control_inputs = [0]

# Initialize figures and axes for overlapping plots
fig_u, ax_u = plt.subplots()
fig_x, ax_x = plt.subplots()
fig_th1, ax_th1 = plt.subplots()
fig_th2, ax_th2 = plt.subplots()
fig_xdot, ax_xdot = plt.subplots()
fig_thdot, ax_thdot = plt.subplots()

for i in range(20, 41, 10):

    y = np.array([0, -0.125, 0.125, 0, 0, 0])
    # y = np.array([1, 0, 0, 0, 0, 0])
    states = [y]
    # print(y.shape)

    #constraints
    x_ref = np.array([0,math.radians(0),-math.radians(0),0,0,0]).reshape(-1,1)
    u_ref = 0

    theta_const = 0.174
    x_const = 3
    u_const = 100
    o = 10

    x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
    x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
    u_ub = np.array(u_const).reshape(-1,1)
    u_lb = np.array(-u_const).reshape(-1,1)

    control_inputs = [0]


    N = i  
    print(f"Running simulation for N = {N}")
    flag = True
    states = []  # reset for each N
    control_inputs = []

    # Initial state setup (assumes you have y and states defined before loop)
    states.append(y.copy())

    # Simulation loop
    for t in time:
        x_cur = states[-1].reshape(-1,1)
        u = mpc_solve(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref)
        if u is None:
            print("MPC not possible for the given conditions")
            flag = False
            break

        print(f"Percentage done: {(t/T)*100:.1f}%")
        y = rk4_step(y, dt, params, u[0][0][0])
        control_inputs.append(u[0][0][0])
        states.append(y)

    if flag:
        states = np.array(states)

        # Optional: animation per N
        # animated(states, time, params, control_inputs, path_out_dir / f"MPC_N{N}.gif")

        # Overlapping plots
        ax_u.step(time, control_inputs, label=f"N={N}")
        ax_u.set_xlabel("Time [s]")
        ax_u.set_ylabel("Control Input [u]")
        ax_u.grid(True)

        ax_x.step(time, states[1:, 0], label=f"N={N}")
        ax_x.set_xlabel("Time [s]")
        ax_x.set_ylabel("Position [x]")
        ax_x.grid(True)

        ax_th1.step(time, np.rad2deg(states[1:, 1]), label=f"Theta1 N={N}")
        ax_th1.set_xlabel("Time [s]")
        ax_th1.set_ylabel("Theta1 [deg]")
        ax_th1.grid(True)

        ax_th2.step(time, np.rad2deg(states[1:, 2]), label=f"Theta2 N={N}")
        ax_th2.set_xlabel("Time [s]")
        ax_th2.set_ylabel("Theta2 [deg]")
        ax_th2.grid(True)

        ax_xdot.step(time, states[1:, 3], label=f"N={N}")
        ax_xdot.set_xlabel("Time [s]")
        ax_xdot.set_ylabel("Velocity [xdot]")
        ax_xdot.grid(True)

        ax_thdot.plot(time, np.rad2deg(states[1:, 4]), label=f"Theta1_dot N={N}")
        ax_thdot.plot(time, np.rad2deg(states[1:, 5]), label=f"Theta2_dot N={N}")
        ax_thdot.set_xlabel("Time [s]")
        ax_thdot.set_ylabel("Angular Velocity [deg/s]")
        ax_thdot.grid(True)

# Finalize and save all figures
ax_u.set_title("Control Input")
ax_u.legend()
fig_u.savefig(path_out_dir / "Control_input_all_N.png")

ax_x.set_title("Position")
ax_x.legend()
fig_x.savefig(path_out_dir / "Position_all_N.png")

ax_th1.set_title("Theta1")
ax_th1.legend()
fig_th1.savefig(path_out_dir / "theta1_all_N.png")

ax_th2.set_title("Theta2")
ax_th2.legend()
fig_th2.savefig(path_out_dir / "theta2_all_N.png")

ax_xdot.set_title("Position_dot")
ax_xdot.legend()
fig_xdot.savefig(path_out_dir / "Position_dot_all_N.png")

ax_thdot.set_title("Theta1_dot and Theta2_dot")
ax_thdot.legend()
fig_thdot.savefig(path_out_dir / "theta_dot_all_N.png")