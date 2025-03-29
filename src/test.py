import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import control as ct

# Load parameters from JSON file
with open("config.json", "r") as file:
    params = json.load(file)

# Defining constants
M, m1, m2 = params["M"], params["m1"], params["m2"]
L1, L2 = params["L1"], params["L2"]
l1, l2 = params["l1"], params["l2"]
I1, I2 = params["I1"], params["I2"]
g = params["g"]

# Defining matrices for the linearized system
J = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, M + m1 + m2, (m1 * l1 + m2 * L1), m2 * l2],
              [0, 0, 0, (m1 * l1 + m2 * L1), m1 * (l1 ** 2) + m2 * (L1 ** 2) + I1, m2 * L1 * l2],
              [0, 0, 0, m2 * l2, m2 * L1 * l2, m2 * (l2 ** 2) + I2]])

N = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0],
              [0, (m1 * l1 + m2 * L1) * g, 0, 0, 0, 0],
              [0, 0, m2 * l2 * g, 0, 0, 0]])

F = np.array([[0],
              [0],
              [0],
              [1],
              [0],
              [0]])

# Continuous dynamic matrices
Ac = np.linalg.inv(J) @ N
Bc = np.linalg.inv(J) @ F

Cc = np.array([1,0,1,0,0,0])
print(Cc)
Dc = np.array([0])

# Create state-space system
sys = ct.ss(Ac, Bc, Cc, Dc)

# Generate Bode plot
plt.figure()
ct.bode(sys)
plt.show()

gm, pm, wcg, wcp = ct.margin(sys)

print(f"Gain Margin: {gm:.2f}, at ω = {wcg:.2f} rad/s")
print(f"Phase Margin: {pm:.2f}°, at ω = {wcp:.2f} rad/s")
