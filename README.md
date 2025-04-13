MPC Project : Double Inverted Pendulum on a cart 

This project attempts to find the optimal control for a Double inverted pendulum with one end fixed to a moving cart using MPC.

Problem Description:
The double inverted pendulums are only allowed to rotate around one axis and the cart is restricted to motion in only one direction. An optimal control for the force applied to the cart has to be found for which the double inverted pendulum is able to stay at it's unstable equilibrium (i.e. upside down).

Main Functions Files:
1. Regulation MPC: regulation_mpc_FINAL.py
2. Output-Free MPC: offset_free_mpc.py
3. Stability analysis: stability_analysis.py
4. config.json holds the configuration for the chosen system

Libraries required:
1. cvxpy
2. numpy 
3. quadprog 
4. scipy 
5. matplotlib
5. control
6. GUROBI optimizer 
7. json
8. pathlib 
9. math



