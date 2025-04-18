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
from ellipsoidal_method import find_maximal_terminal_set


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
dt= 0.02

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

Q = sp.linalg.block_diag(1000,100,100,1000,1,1)

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

control_inputs = [0]

#MPC horizon
N = 40
N = int(N)  
print(N)
flag = True
# Simulation loop
for t in time:

    x_cur = states[-1].reshape(-1,1)
    u = mpc_solve(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref)
    if u is None:
        print("MPC not possible for the given conditions")
        flag = False
        break
    print(f"Percentage done",(t/T)*100)
    y = rk4_step(y, dt,params,u[0][0][0])
    control_inputs.append(u[0][0][0])
    states.append(y)

if flag:
    states = np.array(states)
    animated(states, time, params, control_inputs,path_out_dir/"MPC.gif")

    plt.figure()
    plt.plot(control_inputs)
    plt.savefig(path_out_dir/"./Control_input_1.png")


    plt.figure()
    plt.plot(states[:,0])
    plt.savefig(path_out_dir/"./Position_1.png")

    plt.figure()
    plt.plot(np.rad2deg(states[:,1]))
    plt.plot(np.rad2deg(states[:,2]))
    plt.legend(['Theta1','Theta2'])
    plt.savefig(path_out_dir/"./theta_1.png")

    plt.figure()
    plt.plot(states[:,3])
    plt.savefig(path_out_dir/"./Position_dot_1.png")

    plt.figure()
    plt.plot(states[:,4])
    plt.plot(states[:,5])
    plt.legend(['Theta1_dot','Theta2_dot'])
    plt.savefig(path_out_dir/"./theta_dot.png")
