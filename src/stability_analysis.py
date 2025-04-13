import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import control as ct
from mpc_solver import mpc_solve_X
from pathlib import Path
from  ellipsoidal_method import find_maximal_terminal_set, findXN, project_vertices_2d, draw_constraints, solve_lyapunov_discrete
from mpc_solver import mpc_solve
from non_linear_dynamics import rk4_step

def terminal_cost(x, P):
    '''
    Returns terminal cost
    '''
    r = 0.5*(x.T @ P @ x)

    if isinstance(r, np.ndarray):
        return r[0][0]
    else:
        return r

def stage_cost(x, u, Q, R):
    '''
    Returns stage cost
    '''
    r = 0.5*(x.T @ Q @ x) + 0.5*(u.T @ R @ u)

    return r[0][0]


path_in_dir_script = Path(__file__).parent #fold where the main script is
path_out_dir = path_in_dir_script / "../out/Stability_Analysis"
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

c = 0.5 * terminal_vertices[0,:] @ P @ terminal_vertices[0,:].T

print(f"c: {c}")

for i in range(6):
    print  (0.5 * terminal_vertices[i,:] @ P @ terminal_vertices[i,:].T)


'''Showing Lyapunov decrease for an inital condition on the edge of Xf'''

T = 0.5
time = np.arange(0, T, dt)

# y = np.array(terminal_vertices[0,:])
y = np.array([0,0.001,0.001,0,0,0])
states = [y]


#constraints
x_ref = np.array([0,0,0,0,0,0]).reshape(-1,1)
u_ref = 0

control_inputs = [0]

lyap_diff = []
stage = []

#MPC horizon
N = 40
N = int(N)  
print(N)
flag = True
# Simulation loop
for t in time:
    x_cur = states[-1].reshape(-1,1)
    # u = mpc_solve(A, B, Q, R, P, x_cur, N, x_lb, x_ub, u_lb, u_ub, x_ref, u_ref)
    # if u is None:
    #     print("MPC not possible for the given conditions")
    #     flag = False
    #     break
    # print(f"Percentage done",(t/T)*100)
    u = K @ x_cur
    y = rk4_step(y, dt,params,u[0][0])
    states.append(y)
    lyap_diff.append(terminal_cost(y.T, P) - terminal_cost(x_cur, P))
    stage.append(stage_cost(x_cur, u[0], Q, R))

states = np.array(states)
lyap_diff = np.array(lyap_diff)
stage = np.array(stage)

plt.step(time,lyap_diff,linestyle='--')
plt.step(time,-stage,linestyle=':')
plt.legend(['V_f(f(x,u)) - V_f(x)','-l(x,u)'])
plt.title("Lyapunov Decrease")
plt.xlabel('Time')
plt.savefig(path_out_dir/"Lyapunov Decrease")

X1, cc = findXN(A, B, K, P, Q, R, x_ub, x_lb, u_ub, u_lb, c, scaling_factor, 100)

'''Showing that the terminal set is a sub-set of X (set of constraints)'''

fig1, ax1 = project_vertices_2d(terminal_vertices, state_indices=[1,2], color = 'blue', hash = '//', title='Terminal Set (Xf) in Set of Contraints(X)')  
draw_constraints(x_ub, x_lb, ax1, state_indices=[1,2])
ax1.legend(['Xf','X'])
fig1.savefig(path_out_dir/'Terminal Set inside Constraints - Theta')

fig2, ax2 = project_vertices_2d(terminal_vertices, state_indices=[0,3], color = 'blue', hash = '//', title='Terminal Set (Xf) in Set of Contraints(X)')  
draw_constraints(x_ub, x_lb, ax2, state_indices=[0,3])
ax2.legend(['Xf','X'])
fig2.savefig(path_out_dir/'Terminal Set inside Constraints - Position')

fig3, ax3 = project_vertices_2d(terminal_vertices, state_indices=[4,5], color = 'blue', hash = '//', title='Terminal Set (Xf) in Set of Contraints(X)')  
draw_constraints(x_ub, x_lb, ax3, state_indices=[4,5])
ax3.legend(['Xf','X'])
fig3.savefig(path_out_dir/'Terminal Set inside Constraints - Omega')

'''Computing XN and overlapping with Xf'''

fig4, ax4 = project_vertices_2d(terminal_vertices, state_indices=[1,2], color = 'blue', hash = 'x', title='Controllable Set for N=100') 
fig4, ax4 = project_vertices_2d(X1, state_indices=[1,2], color = 'orange',hash = '-', title=None, ax=ax4) 
ax4.legend(['Xf','X_N'])
fig4.savefig(path_out_dir/'Control Invariant Set X_N - Theta')

fig5, ax5 = project_vertices_2d(terminal_vertices, state_indices=[0,3], color = 'blue', hash = 'x', title='Controllable Set for N=100') 
fig5, ax5 = project_vertices_2d(X1, state_indices=[0,3], color = 'orange',hash = '-', title=None, ax=ax5) 
ax5.legend(['Xf','X_N'])
fig5.savefig(path_out_dir/'Control Invariant Set X_N - Position')

fig6, ax6 = project_vertices_2d(terminal_vertices, state_indices=[4,5], color = 'blue', hash = 'x', title='Controllable Set for N=100') 
fig6, ax6 = project_vertices_2d(X1, state_indices=[4,5], color = 'orange',hash = '-', title=None, ax=ax6) 
ax6.legend(['Xf','X_N'])
fig6.savefig(path_out_dir/'Control Invariant Set X_N - Omega')


