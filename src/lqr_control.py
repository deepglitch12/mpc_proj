import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import dynamics,rk4_step
from draw import animated
import math
from pathlib import Path


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/LQR"
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
# print(np.real(np.linalg.eigvals(Ac)))
dt= 0.01
print(dt)
dsys = sp.signal.cont2discrete((Ac,Bc,Cc,Dc),dt,method='zoh')

A = dsys[0]
B = dsys[1]
C = dsys[2]
D = dsys[3]

Q = sp.linalg.block_diag(1000,100,100,1,1,1)

R = 100

K,S,E = ct.dlqr(A,B,Q,R)

T = 10
time = np.arange(0, T, dt)

y = np.array([0, -0.125, 0.125, 0, 0, 0])
states = [y]
control_inputs = [0]

for t in time:
    u = - K @ y
    y = rk4_step(y, dt,params,u[0])
    states.append(y)
    control_inputs.append(u[0])

states = np.array(states)
animated(states, time, params, control_inputs,path_out_dir/"./Simualtion_lqr.gif")

plt.figure()
plt.plot(control_inputs)
plt.savefig(path_out_dir/"./Control_input_lqr.png")

plt.figure()
plt.plot(states[:,0])
plt.savefig(path_out_dir/"./Position_lqr.png")

plt.figure()
plt.plot(np.rad2deg(states[:,1]))
plt.plot(np.rad2deg(states[:,2]))
plt.legend(['Theta_1',"Theta_2"])
plt.savefig(path_out_dir/"./theta_lqr.png")

plt.figure()
plt.plot(states[:,3])
plt.savefig(path_out_dir/"./Position_dot_lqr.png")

plt.figure()
plt.plot(states[:,4])
plt.plot(states[:,5])
plt.legend(['Theta_1_dot',"Theta_2_dot"])
plt.savefig(path_out_dir/"./theta_dot_lqr.png")