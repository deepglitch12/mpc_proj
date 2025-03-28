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
# print(np.real(np.linalg.eigvals(Ac)))
dt= 0.01
print(dt)
dsys = sp.signal.cont2discrete((Ac,Bc,Cc,Dc),dt,method='zoh')

A = dsys[0]
B = dsys[1]
C = dsys[2]
D = dsys[3]

# print(f"Eigen Values of A:",np.linalg.eigvals(A))
# print(f"Dominant Eigen Value:",min(np.real(A)))


# LQR Control
#defining the weights

print(f"Rank of Controllability Matrix:",np.linalg.matrix_rank(ct.ctrb(A,B)))

Q = sp.linalg.block_diag(100,100,100,1,1,1)

R = 1

P = np.eye(A.shape[0])

T = 10
time = np.arange(0, T, dt)

y = np.array([0, np.pi/20, -np.pi/20, 0, 0, 0])
states = [y]
# print(y.shape)

#constraints

theta_const = math.radians(15)
x_const = 3

x_ub = np.array([x_const, theta_const, theta_const, float('inf'),float('inf'),float('inf')])
x_lb = np.array([-x_const,-theta_const,-theta_const,float('-inf'),float('-inf'),float('-inf')])
u_ub = float('inf') 
u_lb = float('-inf') 

control_inputs = [0]

#MPC horizon
N = (T/dt)*0.20
N = int(N)  
print(N)
# Simulation loop
for t in time:
    u = mpc_solve(A, B, Q, R, P, states[-1], N, x_lb, x_ub, u_lb, u_ub)
    print(f"Percentage done",(t/T)*100)
    y = rk4_step(y, dt,params,u[0][0])
    control_inputs.append(u[0][0])
    print(f"Position:",y[0])
    print(f"Theta1:",y[1])
    print(f"Theta2:",y[2])
    states.append(y)

states = np.array(states)
animated(states, time, params, control_inputs)


plt.figure()
plt.plot(control_inputs)
plt.savefig("Control_input.png")

plt.figure()
plt.plot(states[:,0])
plt.savefig("Position.png")

plt.figure()
plt.plot(states[:,1])
plt.savefig("theta1.png")

plt.figure()
plt.plot(states[:,2])
plt.savefig("theta2.png")