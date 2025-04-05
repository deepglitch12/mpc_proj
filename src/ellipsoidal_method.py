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
from scipy.linalg import solve
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse

def terminal_cost(x, P):
    r = 0.5*(x.T @ P @ x)
    return r[0][0]

def stage_cost(x, u, Q, R):
    r = 0.5*(x.T @ Q @ x) + 0.5*(u.T @ R @ u)
    return r[0][0]

def eigen_polyhedron(P, scale=1.0):
    """
    Constructs a polyhedron from eigenvectors of P, properly scaled
    
    Parameters:
    P : numpy.ndarray - positive definite matrix
    scale : float - initial scaling factor
    
    Returns:
    vertices : numpy.ndarray - vertices of the polyhedron
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    # Get real parts (in case of complex eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Normalize and scale eigenvectors
    scaled_eigenvectors = np.zeros_like(eigenvectors)
    for i in range(eigenvectors.shape[1]):
        scaled_eigenvectors[:,i] = (eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])) * scale
    
    # Create vertices by combining positive and negative eigenvectors
    vertices = np.vstack([scaled_eigenvectors, -scaled_eigenvectors])
    
    return vertices

def find_maximal_terminal_set(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb, max_iter=1000, tol=1e-3):
    """
    Finds the maximal terminal set satisfying all constraints
    
    Parameters:
    A, B, K, P, Q, R : system matrices
    x_ub, x_lb : state constraints
    u_ub, u_lb : input constraints
    max_iter : maximum iterations
    tol : tolerance for convergence
    
    Returns:
    V : vertices of the terminal set
    c : scaling factor
    """
    # Initial scaling factor
    c = 1.0
    best_c = 0.0
    best_V = None
    
    for _ in range(max_iter):
        V = eigen_polyhedron(P, c)
        all_satisfied = True
        
        for v in V:
            v = v.reshape(-1, 1)
            
            # Check state constraints
            if np.any(v > x_ub) or np.any(v < x_lb):
                all_satisfied = False
                break
                
            # Check input constraints
            u = K @ v
            if np.any(u > u_ub) or np.any(u < u_lb):
                all_satisfied = False
                break
                
            # Check Lyapunov decrease condition
            v_next = A @ v + B @ (K @ v)
            current_cost = terminal_cost(v, P)
            next_cost = terminal_cost(v_next, P)
            stage_cost_val = stage_cost(v, K@v, Q, R)
            
            if not (next_cost <= current_cost - stage_cost_val + tol):
                all_satisfied = False
                break
        
        if all_satisfied:
            best_c = c
            best_V = V
            c *= 1.1  # Try larger set
        else:
            c *= 0.9  # Reduce set size
            
        if abs(c - best_c) < tol:
            break
    
    return best_V, best_c


def solve_lyapunov_discrete(A, Q):
    n = A.shape[0]
    # Create identity matrix of size n²
    I = np.eye(n)
    
    # Compute Kronecker product A ⊗ A
    kron_AA = np.kron(A.T, A.T)
    
    # Reshape the equation to (Aᵀ ⊗ Aᵀ - I) vec(P) = -2 vec(Q)
    left_side = kron_AA - np.eye(n*n)
    right_side =  -Q.reshape(-1, 1)
    
    # Solve the linear system
    vec_P = np.linalg.solve(left_side, right_side)
    
    # Reshape the solution back to matrix form
    P = vec_P.reshape(n, n)
    
    return P
    


def get_minimum_volume_ellipse(points, tolerance=0.01):
    """
    Find the minimum volume enclosing ellipse around a set of points.
    Uses the Khachiyan algorithm.
    
    Parameters:
    points : (N, 2) numpy array of points
    tolerance : convergence tolerance
    
    Returns:
    center : (2,) array, ellipse center
    A : (2,2) array, ellipse matrix
    """
    N, d = points.shape
    Q = np.vstack([points.T, np.ones(N)]).T
    u = np.ones(N)/N
    
    while True:
        X = Q * u[:, np.newaxis]
        M = np.dot(X.T, Q)
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break
            
        # Compute new u
        j = np.diag(np.dot(Q, np.dot(inv_M, Q.T)))
        max_j = np.argmax(j)
        step_size = (j[max_j] - d - 1.0) / ((d + 1) * (j[max_j] - 1.0))
        new_u = (1 - step_size) * u
        new_u[max_j] += step_size
        
        # Check convergence
        if np.linalg.norm(new_u - u) < tolerance:
            break
        u = new_u
    
    # Center and matrix
    center = np.dot(u, points)
    A = np.linalg.inv(np.dot(points.T, np.dot(np.diag(u), points)) - 
                      np.outer(center, center)) / d
    
    return center, A

def project_vertices_2d(vertices, state_indices=[0, 1], ax=None, draw_hull=True, draw_ellipse=True):
    """
    Projects vertices onto a 2D plane defined by two state variables and draws
    either the convex hull, an ellipse that tightly fits the projected shape.
    
    Parameters:
    vertices : numpy.ndarray - (n_vertices, n_states) array of vertices
    state_indices : list - indices of two states to project onto [x_index, y_index]
    ax : matplotlib axis - optional axis to plot on
    draw_hull : bool - whether to draw the convex hull
    draw_ellipse : bool - whether to draw an ellipse
    
    Returns:
    fig, ax : figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Extract the two dimensions of interest
    proj_vertices = vertices[:, state_indices]

    # Plot projected points
    ax.scatter(proj_vertices[:, 0], proj_vertices[:, 1], color='red', s=50, label='Vertices')

    if draw_hull:
        hull = ConvexHull(proj_vertices)
        for simplex in hull.simplices:
            ax.plot(proj_vertices[simplex, 0], proj_vertices[simplex, 1], 'k-', lw=2)
        ax.plot(proj_vertices[hull.vertices[[0, -1]], 0], proj_vertices[hull.vertices[[0, -1]], 1], 'k-', lw=2)

    if draw_ellipse:
        # Get convex hull points
        hull_points = proj_vertices[hull.vertices]
        
        # Compute minimum area ellipse
        center, A = get_minimum_volume_ellipse(hull_points)
        
        # Get ellipse parameters
        U, s, V = np.linalg.svd(A)
        width, height = 2/np.sqrt(s)
        angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
        
        # Create ellipse
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                         edgecolor='blue', facecolor='none', lw=2, label='Min Area Ellipse')
        ax.add_patch(ellipse)

    # Axis labels
    state_names = ['Position', 'Angle 1', 'Angle 2', 'Velocity', 'AngVel 1', 'AngVel 2']
    ax.set_xlabel(f'State {state_indices[0]} ({state_names[state_indices[0]]})')
    ax.set_ylabel(f'State {state_indices[1]} ({state_names[state_indices[1]]})')
    ax.set_title('2D Projection of Terminal Set')
    ax.grid(True)
    ax.legend()
    
    return fig, ax


if __name__ == '__main__':
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

    #Q and R Matrices
    Q = sp.linalg.block_diag(1000,100,100,1,1,1) #having no weight on the disturbance component

    R = 100*np.eye(B.shape[1])

    K,_,_ = ct.dlqr(A,B,Q,R)

    K = -K

    Ak = A + B @ K

    if max(np.linalg.eigvals(Ak)>1):
        print('Closed Loop Unstable')
    else:
        print("Closed Loop eigen values inside unit circle i.e. stable")

    Qk = Q + K.T @ R @ K

    P = solve_lyapunov_discrete(Ak,Qk) #Solution to equation given in slides ... sus!!!

    theta_const = 0.174 #10 degrees in radians
    x_const = 3
    u_const = 100
    o = 5

    x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
    x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
    u_ub = np.array(u_const).reshape(-1,1)
    u_lb = np.array(-u_const).reshape(-1,1)

    terminal_vertices, scaling_factor = find_maximal_terminal_set(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb)

    print(f"Found terminal set with scaling factor: {scaling_factor}")

    fig1, ax1 = project_vertices_2d(terminal_vertices, state_indices=[1,2])  # Position vs Angle 1
    plt.show()