import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ==========================================
# PART 1: THE CORE NUMERICAL SOLVER (Q1 & Q2)
# ==========================================
def solve_wave(N, M, L=1.0, T_end=2.0, c=1.0, is_q3=True):
    """
    Solves the 1D wave equation using the Implicit Finite Difference Scheme.
    If is_q3=True, it uses the specific forcing function f(x,t) from Question 3.
    If is_q3=False, it simulates a standard unforced string (f=0) for Question 5.
    """
    dx = L / N
    dt = T_end / M
    r = (c * dt / dx) ** 2

    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T_end, M + 1)

    # Initialize displacement matrix U (rows=space, cols=time)
    U = np.zeros((N + 1, M + 1))

    # Forcing function definition
    def f(x_val, t_val):
        if is_q3:
            return -3 * np.pi ** 2 * np.sin(np.pi * x_val) * np.cos(2 * np.pi * t_val)
        return np.zeros_like(x_val)

    # Initial Conditions setup
    U[:, 0] = np.sin(np.pi * x)
    U[:, 1] = np.copy(U[:, 0])  # Initial velocity u_t(x,0) = 0

    # Tridiagonal Matrix Formulation (Question 2)
    main_diag = (1 + 2 * r) * np.ones(N - 1)
    off_diag = -r * np.ones(N - 2)
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    # Pre-compute LU Decomposition for massive simulation speedup
    P, L_mat, U_mat = la.lu(A)

    # Time-stepping loop
    for n in range(1, M):
        U_current = U[1:N, n]
        U_prev = U[1:N, n - 1]
        F_n = f(x[1:N], t[n])

        # Right hand side of the equation
        b = 2 * U_current - U_prev + (dt ** 2) * F_n

        # Solve AU_next = b using LU decomposition
        y = la.solve_triangular(L_mat, np.dot(P.T, b), lower=True)
        U[1:N, n + 1] = la.solve_triangular(U_mat, y, lower=False)

    return x, t, U, dx, dt


# ==========================================
# PART 2: ENERGY COMPUTATION (Q4)
# ==========================================
def compute_energy(U, dx, dt, c):
    """
    Approximates numerical derivatives (u_t, u_x) and uses
    the Composite Simpson's Rule to calculate total mechanical energy over time.
    """
    # Calculate numerical derivatives using central differences
    u_x = np.gradient(U, dx, axis=0)
    u_t = np.gradient(U, dt, axis=1)

    energy = np.zeros(U.shape[1])

    # Integrate using Composite Simpson's Rule over the spatial domain
    for n in range(U.shape[1]):
        integrand = u_t[:, n] ** 2 + (c ** 2) * u_x[:, n] ** 2
        energy[n] = 0.5 * integrate.simpson(integrand, dx=dx)

    return energy


# ==========================================
# EXECUTION SCRIPT (Runs Q3, Q4, and Q5)
# ==========================================
if __name__ == "__main__":

    # ------------------------------------------
    # QUESTION 3: Simulation, Plotting, and Error
    # ------------------------------------------
    print("--- Running Question 3 ---")
    N_test, M_test = 50, 100
    x, t, U, dx, dt = solve_wave(N_test, M_test, is_q3=True)
    exact_final = np.sin(np.pi * x) * np.cos(2 * np.pi * 2.0)

    # Q3.2 & Q3.4: Plot static waves vs Exact Solution
    plt.figure(figsize=(10, 6))
    times_to_plot = [0, int(M_test / 4), int(M_test / 2), M_test]
    for n in times_to_plot:
        plt.plot(x, U[:, n], 'o', markersize=4, label=f'Numerical t={t[n]:.1f}')
    plt.plot(x, exact_final, 'k--', linewidth=2, label='Exact t=2.0')
    plt.title("Q3.2 & Q3.4: Numerical vs Exact Wave Displacement")
    plt.xlabel("Position (x)")
    plt.ylabel("Displacement (u)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Q3.3: Animation (Saved as GIF)
    print("Generating Animation for Q3.3...")
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(x, U[:, 0], 'b', lw=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Q3.3: Vibrating String Animation (Standing Wave)")
    ax.grid(True)


    def update(frame):
        line.set_ydata(U[:, frame])
        return line,


    ani = FuncAnimation(fig, update, frames=range(0, M_test, 2), blit=True)
    ani.save("wave_animation.gif", writer='pillow', fps=30)
    plt.close()
    print("Animation saved as 'wave_animation.gif'")

    # Q3.5: Convergence Analysis (Log-Log Scale)
    print("Calculating Error Convergence...")
    N_values = [10, 20, 40, 80, 160]
    E_inf_list, E_2_list, dx_list = [], [], []

    for N_val in N_values:
        M_val = N_val * 2  # Keep the ratio constant for stability
        x_grid, t_grid, U_res, dx_val, _ = solve_wave(N_val, M_val, is_q3=True)
        num_final = U_res[:, -1]
        exact_res = np.sin(np.pi * x_grid) * np.cos(2 * np.pi * 2.0)

        E_inf = np.max(np.abs(exact_res - num_final))
        E_2 = np.sqrt(dx_val * np.sum((exact_res - num_final) ** 2))

        E_inf_list.append(E_inf)
        E_2_list.append(E_2)
        dx_list.append(dx_val)

    plt.figure(figsize=(8, 6))
    plt.loglog(dx_list, E_inf_list, 's-', label=r'$E_\infty$ (Max Error)', linewidth=2)
    plt.loglog(dx_list, E_2_list, 'o-', label=r'$E_2$ ($L^2$ Error)', linewidth=2)
    plt.title("Q3.5: Convergence Analysis (Log-Log Scale)")
    plt.xlabel(r"Spatial Step Size $\Delta x$ (Log Scale)")
    plt.ylabel("Error at Final Time T (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    # ------------------------------------------
    # QUESTION 4: Energy Computation
    # ------------------------------------------
    print("\n--- Running Question 4 ---")
    energy = compute_energy(U, dx, dt, c=1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(t, energy, 'r-', linewidth=2)
    plt.title("Q4: Total Mechanical Energy Over Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Energy E(t)")
    plt.grid(True)
    plt.show()

    # ------------------------------------------
    # QUESTION 5: Wave Speed Identification
    # ------------------------------------------
    print("\n--- Running Question 5 ---")

    t_star = 0.5
    u_obs = 0.35


    # Root finding function: F(c) = u(L/2, t*; c) - u_obs = 0
    def F(c_guess):
        # We run a lower-resolution simulation for faster root finding
        x_g, t_g, U_g, _, _ = solve_wave(20, 40, c=c_guess, is_q3=False)
        mid_x_idx = len(x_g) // 2
        t_star_idx = int((t_star / 2.0) * 40)
        return U_g[mid_x_idx, t_star_idx] - u_obs


    # Q5.1: Bisection Method
    print("Executing Bisection Method (Bracket: [0.1, 2.0])...")
    result_bisect = root_scalar(F, bracket=[0.1, 2.0], method='bisect', xtol=1e-3)
    c_approx = result_bisect.root
    print(f"-> Bisection Guess for c: {c_approx:.4f}")

    # Q5.2: Newton's Method (using Bisection result as starting point)
    print("Executing Newton's Method to refine the guess...")
    # Approximating the derivative F'(c) numerically
    fprime_approx = lambda c_val: (F(c_val + 0.001) - F(c_val)) / 0.001

    result_newton = root_scalar(F, x0=c_approx, fprime=fprime_approx, method='newton')
    c_exact = result_newton.root
    print(f"-> Newton's Refined Wave Speed c: {c_exact:.6f}")

    print("\nAll project requirements completed successfully.")