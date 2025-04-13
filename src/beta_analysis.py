import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import dynamics,rk4_step
from draw import animated
from mpc_solver import mpc_solve_beta
import math
from pathlib import Path


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/Beta_Analysis"
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

# LQR Control
#defining the weights

print(f"Rank of Controllability Matrix:",np.linalg.matrix_rank(ct.ctrb(A,B)))

Q = sp.linalg.block_diag(1000,100,100,1,1,1)

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
o = 10

x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
u_ub = np.array(u_const).reshape(-1,1)
u_lb = np.array(-u_const).reshape(-1,1)

#MPC horizon
N = 40
N = int(N)  
print(N)
flag = True
# Simulation loop
fig_u, ax_u = plt.subplots(figsize=(8, 6))
ax_u.set_ylabel("Control Input")
ax_u.set_xlabel("Time")


fig_x, ax_x = plt.subplots(figsize=(8, 6))
ax_x.set_ylabel("Position (m)")
ax_x.set_xlabel("Time")


fig_th1, ax_th1 = plt.subplots(figsize=(8, 6))
ax_th1.set_ylabel("Theta_1 (degrees)")
ax_th1.set_xlabel("Time")


fig_th2, ax_th2 = plt.subplots(figsize=(8, 6))
ax_th2.set_ylabel("Theta_2 (degrees)")
ax_th2.set_xlabel("Time")


fig_v, ax_v = plt.subplots(figsize=(8, 6))
ax_v.set_ylabel("Velocity (m/s)")
ax_v.set_xlabel("Time")


fig_w, ax_w = plt.subplots(figsize=(8, 6))
ax_w.set_ylabel("Angular Velocity (rad/s)")
ax_w.set_xlabel("Time")
ax_w.legend(['Theta1_dot','Theta2_dot'])


beta_test = [0.1, 1, 10, 100]
for beta in beta_test:

    y = np.array([0, -0.125, 0.125, 0, 0, 0])
    states = [y]
    control_inputs = [0]
    for t in time:

        x_cur = states[-1].reshape(-1,1)
        u = mpc_solve_beta(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref,beta)
        if u is None:
            print("MPC not possible for the given conditions")
            flag = False
            break
        print(f"On Beta:",beta)
        print(f"Percentage done",(t/T)*100)
        y = rk4_step(y, dt,params,u[0][0][0])
        control_inputs.append(u[0][0][0])
        states.append(y)

    if flag:
        states = np.array(states)

        ax_w.step(time, states[1:,4])
        ax_w.step(time, states[1:,5])
        ax_v.step(time, states[1:,3])
        ax_x.step(time, states[1:,0])
        ax_u.step(time, control_inputs[1:])
        ax_th1.step(time ,np.rad2deg(states[1:,1]))
        ax_th2.step(time, np.rad2deg(states[1:,2]))

        

ax_w.legend(beta_test)
ax_w.set_title("Theta_dot")
fig_w.savefig(path_out_dir/f"./theta_dot.png")

ax_v.set_title("Velocity")
fig_v.savefig(path_out_dir/f"./Position_dot.png")

ax_th2.legend(beta_test)
ax_th2.set_title("Theta 2")
fig_th2.savefig(path_out_dir/f"./theta_2.png")

ax_th1.legend(beta_test)
ax_th1.set_title("Theta 1")
fig_th1.savefig(path_out_dir/f"./theta_1.png")

ax_x.legend(beta_test)
ax_x.set_title("Postion")
fig_x.savefig(path_out_dir/f"./Position.png")

ax_u.legend(beta_test)
ax_u.set_title("Control Input")
fig_u.savefig(path_out_dir/f"./Control_input.png")
