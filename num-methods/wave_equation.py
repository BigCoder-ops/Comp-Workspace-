import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & GRID SETUP
# ==========================================
L = 1.0  # Length of the string
T_end = 2.0  # Final time
N = 50  # Number of spatial intervals
M = 100  # Number of time steps
c = 1.0  # Wave speed

dx = L / N
dt = T_end / M
r = (c * dt / dx) ** 2

x = np.linspace(0, L, N + 1)
t = np.linspace(0, T_end, M + 1)

# Initialize the solution matrix U
U = np.zeros((N + 1, M + 1))


# External force function
def f(x, t):
    return -3 * np.pi ** 2 * np.sin(np.pi * x) * np.cos(2 * np.pi * t)


# ==========================================
# 2. INITIAL CONDITIONS
# ==========================================
# Time step n=0 (t=0): u(x,0) = sin(pi * x)
U[:, 0] = np.sin(np.pi * x)

# Time step n=1 (t=dt): We use the initial velocity u_t(x,0) = 0 to find U[:, 1].
# Using a simple forward difference for the first step: (U[:, 1] - U[:, 0]) / dt = 0
# Therefore, U[:, 1] = U[:, 0]
U[:, 1] = np.copy(U[:, 0])

# ==========================================
# 3. MATRIX SETUP (LU DECOMPOSITION)
# ==========================================
main_diag = (1 + 2 * r) * np.ones(N - 1)
off_diag = -r * np.ones(N - 2)
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

# Pre-compute LU decomposition for speed inside the loop
P, L_mat, U_mat = la.lu(A)

# ==========================================
# 4. TIME-STEPPING LOOP
# ==========================================
for n in range(1, M):
    U_current = U[1:N, n]
    U_prev = U[1:N, n - 1]

    F_n = f(x[1:N], t[n])

    # Construct Right-Hand Side
    b = 2 * U_current - U_prev + (dt ** 2) * F_n

    # Solve A * U_next = b using LU
    y = la.solve_triangular(L_mat, np.dot(P.T, b), lower=True)
    U_next = la.solve_triangular(U_mat, y, lower=False)

    # Store result
    U[1:N, n + 1] = U_next

# ==========================================
# 5. VISUALIZATION (Example Plotting)
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(x, U[:, 0], label='t=0.0')
plt.plot(x, U[:, int(M / 4)], label=f't={T_end / 4}')
plt.plot(x, U[:, int(M / 2)], label=f't={T_end / 2}')
plt.title("Wave Equation: String Displacement Over Time")
plt.xlabel("Position (x)")
plt.ylabel("Displacement (u)")
plt.legend()
plt.grid(True)
plt.show()