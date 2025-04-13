import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import control as ct
from mpc_solver import mpc_solve_X
from pathlib import Path
from matplotlib.patches import Ellipse, Polygon

def terminal_cost(x, P):
    '''
    Returns terminal cost
    '''
    r = 0.5*(x.T @ P @ x)
    return r[0][0]

def stage_cost(x, u, Q, R):
    '''
    Returns stage cost
    '''
    r = 0.5*(x.T @ Q @ x) + 0.5*(u.T @ R @ u)
    return r[0][0]

def eigen_polyhedron(P, c=1.0):
    """
    Constructs a polyhedron aligned with eigenvectors of P that lies inside the ellipse 0.5 x^T P x <= c
    """
    eigenvalues, eigenvectors = np.linalg.eigh(P)  # ensure symmetric P

    vertices = []
    for i in range(P.shape[0]):
        direction = eigenvectors[:, i]
        scale = np.sqrt(2 * c / eigenvalues[i])
        vertices.append(scale * direction)
        vertices.append(-scale * direction)

    return np.array(vertices)  # shape (2n, n)

def eigen_rectangle(P, c=1.0):
    """
    Constructs a rectangle aligned with eigenvectors of P that bounds the ellipse 0.5 x^T P x <= c
    Returns 4 corners of the rectangle (each corner is a 2D point)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(P)  # ensure symmetric P

    # Get the principal directions and corresponding lengths
    axes_lengths = np.sqrt(2 * c / eigenvalues)  # scale factors along eigenvectors

    # 2D vectors along principal axes
    v1 = eigenvectors[:, 0] * axes_lengths[0]
    v2 = eigenvectors[:, 1] * axes_lengths[1]

    # Corners of the rectangle (centered at origin)
    corners = np.array([
        +v1 + v2,
        +v1 - v2,
        -v1 - v2,
        -v1 + v2
    ])

    return corners

def findXN(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb, c, initial_upper, N, tol=1e-3, max_iter=50):
    """
    Finds largest polyhedral set aligned with eigenvectors of P such that all vertices are feasible.
    Uses binary search over the scaling factor.
    """

    lower = initial_upper
    upper = initial_upper*2
    best_scale = 0.0
    best_V = None

    for _ in range(max_iter):
        scale = (upper + lower) / 2.0
        V_check = eigen_rectangle(P, scale)
        all_satisfied = True

        for v in V_check:
            v = v.reshape(-1, 1)

           # Check state constraints
            if np.any(v > x_ub) or np.any(v < x_lb):
                all_satisfied = False
                break
                
            u = mpc_solve_X(A, B, Q, R, P, v, N, x_lb, x_ub, u_lb, u_ub, c)
            if u is None:
                all_satisfied = False
                break

        if all_satisfied:
            best_scale = scale
            lower = scale  # Try bigger set
        else:
            upper = scale  # Too large, reduce

        if abs(upper - lower) < tol:
            break

    return eigen_polyhedron(P,best_scale), best_scale




def find_maximal_terminal_set(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb, max_iter=1000):
    """
    Finds the maximal terminal set satisfying all constraints
    """
    # Initial scaling factor
    c = 1.0
    best_c = 0.0
    best_V = None
    
    for _ in range(max_iter):
        V = eigen_rectangle(P, c)
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

            if not (next_cost < current_cost):
                all_satisfied = False
                break

            if not (next_cost <= current_cost - stage_cost_val):
                all_satisfied = False
                break
        
        if all_satisfied:
            best_c = c
            c *= 1.1  # Try larger set
        else:
            c *= 0.9  # Reduce set size
            
        if abs(c - best_c) < 0.01:
            break
    
    return eigen_polyhedron(P,best_c), best_c


def solve_lyapunov_discrete(A, Q):
    n = A.shape[0]
    # Create identity matrix of size n²
    I = np.eye(n)
    
    # Compute Kronecker product A ⊗ A
    kron_AA = np.kron(A.T, A.T)
    
    # Reshape the equation to (Aᵀ ⊗ Aᵀ - I) vec(P) = -2 vec(Q)
    left_side = kron_AA - np.eye(n*n)
    right_side =  -1*Q.reshape(-1, 1)
    
    # Solve the linear system
    vec_P = np.linalg.solve(left_side, right_side)
    
    # Reshape the solution back to matrix form
    P = vec_P.reshape(n, n)
    
    return P
    

def get_minimum_volume_ellipse(points, tolerance=0.01):
    """
    Compute the minimum volume enclosing ellipse (MVEE) using the Khachiyan algorithm.
    """
    N, d = points.shape
    Q = np.vstack([points.T, np.ones(N)]).T
    u = np.ones(N) / N
    
    while True:
        X = Q * u[:, np.newaxis]
        M = np.dot(X.T, Q)
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break
            
        j = np.diag(np.dot(Q, np.dot(inv_M, Q.T)))
        max_j = np.argmax(j)
        step_size = (j[max_j] - d - 1.0) / ((d + 1) * (j[max_j] - 1.0))
        new_u = (1 - step_size) * u
        new_u[max_j] += step_size
        
        if np.linalg.norm(new_u - u) < tolerance:
            break
        u = new_u
    
    center = np.dot(u, points)
    A = np.linalg.inv(np.dot(points.T, np.dot(np.diag(u), points)) - np.outer(center, center)) / d
    
    return center, A

def project_vertices_2d(vertices, state_indices=[0, 1], ax=None, color='blue',hash=None, title=None):
    """
    Projects vertices onto a 2D plane and draws ONLY the enclosing ellipse.
    
    """
    proj_vertices = vertices[:, state_indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        x_max = 2 * np.max(abs(proj_vertices))
        ax.set_xlim([-x_max,x_max])
        ax.set_ylim([-x_max,x_max])
    else:
        fig = ax.figure
    

    # Compute minimum volume enclosing ellipse
    center, A = get_minimum_volume_ellipse(proj_vertices)

    # for i in range(vertices.shape[0]):
    #     ax.plot(vertices[i,state_indices[0]],vertices[i,state_indices[1]],'*')
    
    # Get ellipse parameters (SVD of shape matrix)
    U, s, V = np.linalg.svd(A)
    width, height = 2 / np.sqrt(s)  # Scaling to match ellipse size
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    
    # Plot the ellipse
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                    
                    edgecolor = color, facecolor = 'none', hatch=hash ,lw=2, label='Enclosing Ellipse')
    ax.add_patch(ellipse)

    # Axis labels and formatting
    state_names = ['Position', 'Angle 1', 'Angle 2', 'Velocity', 'AngVel 1', 'AngVel 2']
    ax.set_xlabel(f'State {state_indices[0]} ({state_names[state_indices[0]]})')
    ax.set_ylabel(f'State {state_indices[1]} ({state_names[state_indices[1]]})')
    if not title is None:
        ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    
    return fig, ax


def draw_constraints(x_ub, x_lb, ax, state_indices):
    i, j = state_indices

    # Extract bounds for the selected state dimensions
    xi_min, xi_max = x_lb[i, 0], x_ub[i, 0]
    xj_min, xj_max = x_lb[j, 0], x_ub[j, 0]

    # Define corners of the polygon (counter-clockwise)
    corners = np.array([
        [xi_min, xj_min],
        [xi_min, xj_max],
        [xi_max, xj_max],
        [xi_max, xj_min]
    ])

    # Create and add polygon to the axis
    poly = Polygon(corners, closed=True, facecolor='none', edgecolor='red', linewidth=1.5, linestyle='-')
    ax.add_patch(poly)

    ax.set_xlim(xi_min-0.1, xi_max+0.1)
    ax.set_ylim(xj_min-0.1, xj_max+0.1)

    

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
    # print(P)
    theta_const = 0.174 #10 degrees in radians
    x_const = 3
    u_const = 100
    o = 10

    x_ub = np.array([x_const, theta_const, theta_const, o, o, o]).reshape(-1,1)
    x_lb = np.array([-x_const,-theta_const,-theta_const,-o,-o,-o]).reshape(-1,1)
    u_ub = np.array(u_const).reshape(-1,1)
    u_lb = np.array(-u_const).reshape(-1,1)

    terminal_vertices, scaling_factor = find_maximal_terminal_set(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb)
    print(f"Found terminal set with scaling factor: {scaling_factor}")
    # print(f"terminal set vertices:",terminal_vertices)

    c = 0.5 * terminal_vertices[0,:] @ P @ terminal_vertices[0,:].T

    print(f"c: {c}")

    X1, cc = findXN(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb, c, scaling_factor, 1)

    fig1, ax1 = project_vertices_2d(terminal_vertices, state_indices=[1,2])  
    fig1 ,ax1 = project_vertices_2d(X1, state_indices=[1,2],ax=ax1,color='purple')
    draw_constraints(x_ub, x_lb, ax1, state_indices=[1,2])

    plt.show()


