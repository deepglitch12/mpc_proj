import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import dynamics,rk4_step
from draw import animated
from mpc_solver import mpc_solve, mpc_solve_beta
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

Q = sp.linalg.block_diag(1000,100,100,1000,10,10)

R = 100

#solution to dare
P ,_,K = ct.dare(A, B, Q, R)

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

control_inputs = [0]

#MPC horizon
N = 40
N = int(N)  
print(N)
beta = 10
flag = True
# Simulation loop
for t in time:

    x_cur = states[-1].reshape(-1,1)
    u = mpc_solve_beta(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref, beta)
    if u is None:
        print("MPC not possible for the given conditions")
        flag = False
        break
    print(f"Percentage done",(t/T)*100)
    y = rk4_step(y, dt,params,u[0][0][0])
    control_inputs.append(u[0][0][0])
    states.append(y)



K_LQR,S_LQR,E_LQR = ct.dlqr(A,B,Q,R)

T = 10
time = np.arange(0, T, dt)

y = np.array([0, -0.125, 0.125, 0, 0, 0])
y_LQR = y
states_LQR = [y_LQR]
control_inputs_LQR = [0]

for t in time:
    u_LQR = - K_LQR @ y_LQR
    y_LQR = rk4_step(y_LQR,dt,params,u_LQR[0])
    states_LQR.append(y_LQR)
    control_inputs_LQR.append(u_LQR[0])

states_LQR = np.array(states_LQR)
#animated(states, time, params, control_inputs,path_out_dir/"./Simualtion_lqr.gif")

states = np.array(states)

# Convert to numpy for easier handling
control_inputs = np.array(control_inputs)
control_inputs_LQR = np.array(control_inputs_LQR)

# Create time axis for plotting
time_axis = np.arange(0, T + dt, dt)

# Plotting comparison of control inputs
plt.figure(figsize=(10, 4))
plt.plot(time_axis, control_inputs, label='MPC')
plt.plot(time_axis, control_inputs_LQR, label='LQR', linestyle='--')
plt.title("Control Inputs Comparison")
plt.xlabel("Time [s]")
plt.ylabel("Control Input [N]")
plt.legend()
plt.grid(True)
plt.savefig(path_out_dir/"./Comparison_Control_Inputs.png")

# Plotting comparison of all states
state_labels = ['x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot']
for i in range(6):
    plt.figure(figsize=(10, 4))
    if i in [1, 2, 4, 5]:  # Convert angles to degrees for better interpretation
        plt.plot(time_axis, np.rad2deg(states[:, i]), label='MPC')
        plt.plot(time_axis, np.rad2deg(states_LQR[:, i]), label='LQR', linestyle='--')
        ylabel = f"{state_labels[i]} [deg]" if 'theta' in state_labels[i] else state_labels[i]
    else:
        plt.plot(time_axis, states[:, i], label='MPC')
        plt.plot(time_axis, states_LQR[:, i], label='LQR', linestyle='--')
        ylabel = state_labels[i]

    plt.title(f"Comparison of {state_labels[i]}")
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(path_out_dir/f"./Comparison_{state_labels[i]}.png")
