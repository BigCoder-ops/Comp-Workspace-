import numpy as np
import matplotlib.pyplot as plt

# Load your data
# year, gdb = np.loadtxt("data.txt", unpack=True)
# Using placeholder data for demonstration
year = np.array([1990, 1991, 1993, 1994, 1995])
gdb = np.array([20.5, 22.1, 28.2, 30.5, 32.8])

def monomial(year, gdb):
    n = len(year)
    V = np.zeros((n, n))
    
    # Fill Vandermonde Matrix: rows = points, cols = powers (0 to n-1)
    for i in range(n):
        for j in range(n):
            V[i, j] = year[i]**j
            
    # Solve V * a = gdb
    coeffs = np.linalg.solve(V, gdb)
    return coeffs # Returns [a0, a1, a2...]

def horner(x, coeffs):
    n = len(coeffs) - 1
    p = coeffs[n] # Start with highest degree coefficient (an)
    
    # Iterate backwards from n-1 down to 0
    for i in range(n - 1, -1, -1):
        p = p * x + coeffs[i]
    return p

def interpolation_graph(year, gdb):
    coeffs = monomial(year, gdb)
    
    # Calculate 1992 GDP
    gdp_1992 = horner(1992, coeffs)
    print(f"Estimated GDP in 1992: {gdp_1992:.6f}")

    # Plotting
    plt.scatter(year, gdb, color='red', label='Data Points')
    
    x_range = np.linspace(min(year), max(year), 100)
    # Use our horner function instead of poly1d to avoid ordering issues
    y_interp = [horner(t, coeffs) for t in x_range]
    
    plt.plot(x_range, y_interp, color='blue', label='Monomial Interpolation')
    plt.axvline(x=1992, color='green', linestyle='--', label='1992 Prediction')
    
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.title('GDP Interpolation using Vandermonde Matrix')
    plt.legend()
    plt.show()

interpolation_graph(year, gdb)