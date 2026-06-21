import numpy as np
import scipy.linalg as la

# 1. Define physical and numerical parameters
L = 1.0          # Length of the string
T_end = 2.0      # Final time
N = 50           # Number of spatial intervals
M = 100          # Number of time steps
c = 1.0          # Wave speed

# 2. Calculate step sizes
dx = L / N
dt = T_end / M

# Calculate 'r' as defined in Question 2
r = (c * dt / dx)**2

# 3. Create the spatial and temporal meshes
x = np.linspace(0, L, N+1)
t = np.linspace(0, T_end, M+1)

# Initialize a 2D array to store the solution u(x, t)
# Rows represent spatial points (x_i), Columns represent time steps (t^n)
U = np.zeros((N+1, M+1))