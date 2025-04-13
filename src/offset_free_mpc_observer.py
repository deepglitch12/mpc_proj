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
from optimal_target import opti_target_online
from scipy.signal import place_poles


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/Offset_free_MPC_Luenberger"
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

#disturbance matrices
Bd = B
Cd = np.array([[1e-4],
               [0],
               [0],
               [0],
               [0],
               [0]])

#checking for Detectability
nd = 1
nx = A.shape[0]
A_aug = np.block([
    [A, Bd],
    [np.zeros((nd, nx)), np.eye(nd)]
])

B_aug = np.vstack([B,np.zeros((1,1))])

C_aug = np.hstack([C,Cd])

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
Q = sp.linalg.block_diag(1000,100,100,1,1,1) #having no weight on the disturbance component

R = 100

#solution to dare
P ,_,K = ct.dare(A, B ,Q, R)

#poles for observer in continous system
poles = np.array([-48, -50, -45, -51, -49, -39, -20])

#poles of observer (placed inside the unit circle)
poles = np.exp(dt*poles)

print(f'Discrete poles:',poles)

L = place_poles(A_aug.T,C_aug.T,poles)
L = L.gain_matrix.T #Creating array

#Disturbance on the system
dist = 1 #constant disturbance to the system

#prior for kalman filter
x_aug = np.array([0, 0, 0, 0, 0, 0, 0]) #augmented states with disturbance

x_aug = x_aug.reshape(-1,1)
states_aug = [x_aug]

y = np.array([1, -math.radians(5), math.radians(5), 0, 0, 0]) #initial sates
states = [y]

#Constraints
theta_const = math.radians(10)
x_const = 3
u_const = 100
o = 5

x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
u_ub = np.array(u_const).reshape(-1,1)
u_lb = np.array(-u_const).reshape(-1,1)


#selecting our desired final state
x_des = np.array([0,math.radians(0),math.radians(0),0,0,0])
u_des = 0
y_ref = np.array([0,math.radians(0),math.radians(0),0,0,0])

y_ref = y_ref.reshape(-1,1)
x_des = np.array(x_des).reshape(-1, 1)
u_des = np.array(u_des).reshape(-1, 1)

control_inputs = [0]

#determining the horizon
N = 100
N = int(N) 

#storing the errors
error = y.reshape(-1,1) - x_aug[:-1]
error = [error]

#storing the optimal target
x_target = []
u_target = []

#Off-set free MPC with observer
for t in time:
    
    print(f"Percentage done",(t/T)*100)
    #determining optimal target
    x_ref ,u_ref = opti_target_online(A, B, C, Bd, Cd, x_aug[-1], y_ref, x_des=x_des, u_des=u_des)
    x_target.append(x_ref)
    u_target.append(u_ref[0])

    print(x_aug)
    #solving the MPC
    u = mpc_solve(A, B, Q, R, P, x_aug[:-1], N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref)

    y = rk4_step_noise(y, dt,params,u[0][0][0],dist, Cd)  #updating states with non-linear dynamics

    x_aug = A_aug @ x_aug + B_aug @ u[0][0].reshape(-1,1) + L @ (y.reshape(-1,1) - C_aug @ x_aug)

    states_aug.append(x_aug)
    control_inputs.append(u[0][0][0])
    states.append(y)
    error.append(y.reshape(-1,1)-x_aug[:-1])

states = np.array(states)
error = np.array(error)
states_aug = np.array(states_aug)
x_target = np.array(x_target)
u_target = np.array(u_target)

animated(states, time, params, control_inputs,path_out_dir/"MPC_offset_free.gif")

#plotting everything
plt.figure()
plt.plot(states_aug[:,6])
plt.savefig(path_out_dir/"./Estimated_Disturbance.png")

plt.figure()
plt.plot(states_aug[:,0])
plt.savefig(path_out_dir/"./Position_offset_free.png")

plt.figure()
plt.plot(np.rad2deg(states_aug[:,1]))
plt.plot(np.rad2deg(states_aug[:,2]))
plt.legend(['Theta1','Theta2'])
plt.savefig(path_out_dir/"./theta_offset_free.png")

plt.figure()
plt.plot(states_aug[:,3])
plt.savefig(path_out_dir/"./Position_dot_offset_free.png")

plt.figure()
plt.plot(states_aug[:,4])
plt.plot(states_aug[:,5])
plt.legend(['Theta1_dot','Theta2_dot'])
plt.savefig(path_out_dir/"./theta_dot_offset_free.png")

plt.figure()
plt.plot(control_inputs)
plt.savefig(path_out_dir/"./Control_input_offset_free.png")

plt.figure()
plt.plot(error[:,0])
plt.savefig(path_out_dir/"./Position_error.png")

plt.figure()
plt.plot(np.rad2deg(error[:,1]))
plt.plot(np.rad2deg(error[:,2]))
plt.legend(['Theta1','Theta2'])
plt.savefig(path_out_dir/"./theta_error.png")

plt.figure()
plt.plot(error[:,3])
plt.savefig(path_out_dir/"./Position_dot_error.png")

plt.figure()
plt.plot(error[:,4])
plt.plot(error[:,5])
plt.legend(['Theta1_dot','Theta2_dot'])
plt.savefig(path_out_dir/"./theta_dot_error.png")

plt.figure()
plt.plot(x_target[:,0])
plt.savefig(path_out_dir/"./Position_target.png")

plt.figure()
plt.plot(np.rad2deg(x_target[:,1]))
plt.plot(np.rad2deg(x_target[:,2]))
plt.legend(['Theta1','Theta2'])
plt.savefig(path_out_dir/"./theta_target.png")

plt.figure()
plt.plot(x_target[:,3])
plt.savefig(path_out_dir/"./Position_dot_target.png")

plt.figure()
plt.plot(x_target[:,4])
plt.plot(x_target[:,5])
plt.legend(['Theta1_dot','Theta2_dot'])
plt.savefig(path_out_dir/"./theta_dot_target.png")

plt.figure()
plt.plot(u_target)
plt.savefig(path_out_dir/"./Control_input_target.png")