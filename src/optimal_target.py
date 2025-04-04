import cvxpy as cp
import numpy as np
from quadprog import solve_qp
import scipy as sp
import matplotlib.pyplot as plt
import scipy as sp
import control as ct
import gurobipy

def opti_target_online(A, B, C, Bd, Cd, d_hat, y_ref, x_des, u_des):

    #defining the weights 
    Q = np.eye(A.shape[0])
    R = np.eye(B.shape[1])

    d_hat = np.array(d_hat).reshape(-1,1)
    
    #defining the optimization variables
    x_ref = cp.Variable((A.shape[0],1))
    u_ref = cp.Variable((B.shape[1],1))

    constraints = []
    constraints += [x_ref  == A @ x_ref + B @ u_ref + Bd @ d_hat]
    constraints += [C @ x_ref == y_ref - Cd @ d_hat]
    
    cost = 0.5 * cp.quad_form(x_des - x_ref, Q) + 0.5 * cp.quad_form(u_des - u_ref, R)
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver = cp.GUROBI,verbose=False)

    return x_ref.value, u_ref.value