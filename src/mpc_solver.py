import cvxpy as cp
import numpy as np
from quadprog import solve_qp
import scipy as sp
import matplotlib.pyplot as plt
import json
import scipy as sp
import control as ct
import gurobipy

def mpc_solve(A, B, Q, R, P, x0, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref):
    
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    # Define the optimization variables
    x_bar = cp.Variable((N+1, dim_x))
    u_bar = cp.Variable((N, dim_u))

    # Cost & constraints
    cost = 0.
    constraints = []
    constraints += [x_bar[0, :] == x0]
    for t in range(N):
        constraints += [x_bar[t+1, :] == A @ x_bar[t, :] + B @ u_bar[t, :]]
        constraints += [x_bar[t+1, :] <= x_ub , x_bar[t+1,:] >= x_lb]
        constraints += [u_lb <= u_bar[t, :], u_bar[t, :] <= u_ub] 

        cost += 0.5 * cp.quad_form(x_bar[t, :] - x_ref, Q) + 0.5 * cp.quad_form(u_bar[t, :] - u_ref, R*np.eye(u_bar[t,:].shape[0]))
    
    cost += 0.5 * cp.quad_form(x_bar[N, :] - x_ref, P)
    
    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver = cp.GUROBI,verbose=False)
    
    return u_bar.value

if __name__ == '__main__':
   
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
    h=0.01
    dsys = sp.signal.cont2discrete((Ac,Bc,Cc,Dc),h,method='zoh')

    A = dsys[0]
    B = dsys[1]
    C = dsys[2]
    D = dsys[3]

    # LQR Control
    #defining the weights

    print(f"Rank of Controllability Matrix:",np.linalg.matrix_rank(ct.ctrb(A,B)))

    Q = sp.linalg.block_diag(100,100,100,1,1,1)

    R = 10

    P = np.eye(A.shape[0])

    dt = 0.1
    T = 10
    time = np.arange(0, T, dt)

    y = np.array([0, np.pi/100, np.pi/100, 0, 0, 0])
    states = [y]

    #constraints
    x_ub = np.array([2,float('inf'),0.1745,float('inf'),0.1745,float('inf')])
    x_lb = np.array([-2,float('-inf'),-0.1745,float('-inf'),-0.1745,float('-inf')])
    u_ub = float('inf')
    u_lb = float('-inf') 

    #MPC horizon
    N = 10         #currently defined as 100 time steps ahead
    u = mpc_solve(A, B, Q, R, P, states[-1], N, x_lb, x_ub, u_lb, u_ub)

    print(u[0][0])