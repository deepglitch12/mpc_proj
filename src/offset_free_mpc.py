import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import rk4_step_noise
from draw import animated
from mpc_solver import mpc_solve
import math
from pathlib import Path


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/Offset_free_MPC"
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

#checking for Controllability
if np.linalg.matrix_rank(ct.ctrb(A,B)) == A.shape[0]:
   print("The System is controllable")
else:
    print("The System is NOT controllable")
    quit()

#checking for Detectability
nd = 1
nx = A.shape[0]
A_aug = np.block([
    [A, B],
    [np.zeros((nd, nx)), np.eye(nd)]
])

B_aug = np.vstack([B,np.zeros((1,1))])

C_aug = np.hstack([C,np.zeros((nx,nd))])

detect_matrix = np.linalg.matrix_rank(np.vstack([np.eye(nx + nd) - A_aug, C_aug]))

print(f"Rank of the detectability matrix: {detect_matrix}")

if detect_matrix < nx + nd:
    print("The System is NOT Detectable")
    quit()
else:
    print("The Modified System is Detectable")

#Duration of the experiment
T = 5
time = np.arange(0, T, dt)

#Q and R Matrices
Q = sp.linalg.block_diag(1000,100,100,1,1,1,0) #having no weight on the disturbance component

R = 100

#solution to dare
P ,_,K = ct.dare(A, B ,Q[:-1,:-1] ,R)

P = sp.linalg.block_diag(P,1)

#Noise covariance for the system
meas_cov = np.diag([1e-4, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
dyn_cov = np.diag([1e-6, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3,0])

#Disturbance on the system
dist = 1 #constant disturbance to the system

#prior for kalman filter
x_aug = np.array([0, 0, 0, 0, 0, 0, 0]) #augmented states with disturbance
P_cov = 2*np.eye(x_aug.shape[0]) #prior covariance

states_aug = [x_aug]

y = np.array([1, -math.radians(5), math.radians(5), 0, 0, 0]) #initial sates
states = [y]

#Constraints
theta_const = math.radians(10)
x_const = 3
u_const = 100

x_ref = np.array([0,math.radians(0),-math.radians(0),0,0,0,0])

x_ub = np.array([x_const, theta_const, theta_const, float('inf'),float('inf'),float('inf'),float('inf')])
x_lb = np.array([-x_const,-theta_const,-theta_const,float('-inf'),float('-inf'),float('-inf'),float('-inf')])
u_ub = u_const
u_lb = -u_const

control_inputs = [0]

N = (T/dt)*0.2
N = int(N) 

error = y - x_aug[:-1]
error = [error]

#Off-set free MPC with Kalman filter
for t in time:

    print(f"Percentage done",(t/T)*100)
    #solving the MPC
    u = mpc_solve(A_aug, B_aug, Q, R, P, x_aug, N, x_lb, x_ub, u_lb, u_ub, x_ref)

    #time update
    x_aug = A_aug @ x_aug + B_aug @ u[0]
    P_cov = A_aug @ P_cov @ A_aug.T + dyn_cov

    #measurement update
    y = rk4_step_noise(y, dt,params,u[0][0],dist,meas_cov)  #updating states with non-linear dynamics

    K_fac = P_cov @ C_aug.T @ np.linalg.inv(C_aug@P_cov@C_aug.T + meas_cov) #Kalman gain
    x_aug = x_aug + K_fac @ (y - C_aug @ x_aug)
    P_cov = P_cov - K_fac @ C_aug @ P_cov

    states_aug.append(x_aug)

    control_inputs.append(u[0][0])
    """debug messages"""
    # print(f"Position:",y[0])
    # print(f"Theta1:",math.degrees(y[1]))
    # print(f"Theta2:",math.degrees(y[2]))
    # print(f'Force:',u[0][0])
    states.append(y)
    error.append(y-x_aug[:-1])

states = np.array(states)
error = np.array(error)
states_aug = np.array(states_aug)
animated(states, time, params, control_inputs,path_out_dir/"MPC_offset_free.gif")

plt.figure()
plt.plot(control_inputs)
plt.savefig(path_out_dir/"./Control_input_offset_free.png")

plt.figure()
plt.plot(states_aug[:,0])
plt.savefig(path_out_dir/"./Position_offset_free.png")

plt.figure()
plt.plot(np.rad2deg(states_aug[:,1]))
plt.savefig(path_out_dir/"./theta1_offset_free.png")

plt.figure()
plt.plot(np.rad2deg(states_aug[:,2]))
plt.savefig(path_out_dir/"./theta2_offset_free.png")

plt.figure()
plt.plot(control_inputs)
plt.savefig(path_out_dir/"./Control_input_offset_free.png")

plt.figure()
plt.plot(error[:,0])
plt.savefig(path_out_dir/"./Position_error.png")

plt.figure()
plt.plot(np.rad2deg(error[:,1]))
plt.savefig(path_out_dir/"./theta1_error.png")

plt.figure()
plt.plot(np.rad2deg(error[:,2]))
plt.savefig(path_out_dir/"./theta2_error.png")