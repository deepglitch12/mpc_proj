import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
from non_linear_dynamics import dynamics,rk4_step
from draw import animated
from mpc_solver import mpc_solve

with open("config.json", "r") as file:
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
dt=0.01
dsys = sp.signal.cont2discrete((Ac,Bc,Cc,Dc),dt,method='zoh')

A = dsys[0]
B = dsys[1]
C = dsys[2]
D = dsys[3]

# LQR Control
#defining the weights

print(f"Rank of Controllability Matrix:",np.linalg.matrix_rank(ct.ctrb(A,B)))

Q = sp.linalg.block_diag(100,10000,10000,1,1,1)

R = 10

P = np.eye(A.shape[0])

T = 10
time = np.arange(0, T, dt)

y = np.array([0, 0, np.pi/100, 0, np.pi/100, 0])
states = [y]
# print(y.shape)

#constraints

theta_const = float('inf')
x_const = float('inf')

x_ub = np.array([x_const,float('inf'),theta_const,float('inf'),theta_const,float('inf')])
x_lb = np.array([-x_const,float('-inf'),-theta_const,float('-inf'),-theta_const,float('-inf')])
u_ub = float('inf') 
u_lb = float('-inf') 


#MPC horizon
N = 10       #currently defined as 100 time steps ahead
# Simulation loop
for t in time:
    u = mpc_solve(A, B, Q, R, P, states[-1], N, x_lb, x_ub, u_lb, u_ub)
    # print(f"On time step",t)
    # print(u[0][0])
    y = rk4_step(y, dt,params,u[0][0])
    states.append(y)

states = np.array(states)
animated(states, time, params)
