import cvxpy as cp
import numpy as np
from quadprog import solve_qp
import scipy as sp
import matplotlib.pyplot as plt
import scipy as sp
import control as ct
import gurobipy

def opti_target_online(A, B, C, d_hat, y_ref, x_des, u_des):

    #defining the weights 
    Q = np.eye(A.shape[0])
    R = 1

    #defining the optimization variables
    x_ref = cp.Variable(A.shape[0])
    u_ref = cp.Variable(B.shape[1])

    constraints = []
    constraints += [x_ref  == A @ (x_ref) + B @ (u_ref + d_hat)]
    # constraints += [C @ (x_ref) == C @ x_des - d_hat]
    
    cost = 0.5 * cp.quad_form(x_des - x_ref, Q) + 0.5 * cp.quad_form(u_des - u_ref, R*np.eye(u_ref.shape[0]))
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver = cp.GUROBI,verbose=False)

    return x_ref.value, u_ref.value