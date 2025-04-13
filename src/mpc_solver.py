import cvxpy as cp
import numpy as np
from quadprog import solve_qp
import scipy as sp
import matplotlib.pyplot as plt
import scipy as sp
import control as ct
import gurobipy
from scipy.optimize import minimize

def mpc_solve(A, B, Q, R, P, x0, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref):
    
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    # Define the optimization variables
    x_bar = cp.Variable((N+1, dim_x, 1))
    u_bar = cp.Variable((N, dim_u, 1))

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
    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except cp.error.SolverError:
    # Solver failed entirely
        return None
    
    return u_bar.value

def mpc_solve_X(A, B, Q, R, P, x0, N, x_lb, x_ub, u_lb, u_ub, c):
    
    dim_x = A.shape[0]
    dim_u = B.shape[1]
    # print(dim_x)
    # Define the optimization variables
    x_bar = cp.Variable((N+1, dim_x, 1))
    u_bar = cp.Variable((N, dim_u, 1))

    # Cost & constraints
    cost = 0.
    constraints = []
    constraints += [x_bar[0, :] == x0]
    for t in range(N):
        constraints += [x_bar[t+1, :] == A @ x_bar[t, :] + B @ u_bar[t, :]]
        constraints += [x_bar[t+1, :] <= x_ub , x_bar[t+1,:] >= x_lb]
        constraints += [u_lb <= u_bar[t, :], u_bar[t, :] <= u_ub] 

        cost += 0.5 * cp.quad_form(x_bar[t, :], Q) + 0.5 * cp.quad_form(u_bar[t, :], R*np.eye(u_bar[t,:].shape[0]))
    
    constraints += [0.5 * cp.quad_form(x_bar[N, :], P) <= c] 
    
    cost += 0.5 * cp.quad_form(x_bar[N, :], P)
    
    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # prob.solve(solver = cp.GUROBI,verbose=False)

    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except cp.error.SolverError:
    # Solver failed entirely
        return None
    
    return u_bar.value


def mpc_solve_beta(A, B, Q, R, P, x0, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref, beta):
    
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    # Define the optimization variables
    x_bar = cp.Variable((N+1, dim_x, 1))
    u_bar = cp.Variable((N, dim_u, 1))

    # Cost & constraints
    cost = 0.
    constraints = []
    constraints += [x_bar[0, :] == x0]
    for t in range(N):
        constraints += [x_bar[t+1, :] == A @ x_bar[t, :] + B @ u_bar[t, :]]
        constraints += [x_bar[t+1, :] <= x_ub , x_bar[t+1,:] >= x_lb]
        constraints += [u_lb <= u_bar[t, :], u_bar[t, :] <= u_ub] 
        

        cost += 0.5 * cp.quad_form(x_bar[t, :] - x_ref, Q) + 0.5 * cp.quad_form(u_bar[t, :] - u_ref, R*np.eye(u_bar[t,:].shape[0]))
    
    cost += 0.5 * beta * cp.quad_form(x_bar[N, :] - x_ref, P)
    
    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except cp.error.SolverError:
    # Solver failed entirely
        return None
    
    return u_bar.value
