import numpy as np
import matplotlib.pyplot as plt

# Load data from text file from the first dataset
year , gdb = np.loadtxt(r"G:\Other computers\Sase-Um6p\Documents\Sase-Docs\Python-Comp\Comp-Workspace-\interpolation\data.txt", unpack=True)

# Load data from text file from the second dataset
#year , gdb = np.loadtxt(r"G:\Other computers\Sase-Um6p\Documents\Sase-Docs\Python-Comp\Comp-Workspace-\interpolation\data2.txt", unpack=True)


def monomial(year, gdb):
    n = len(year)
    V = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            V[i, j] = year[i]**j
            
    
    coeffs = np.linalg.solve(V, gdb)
    return coeffs 
def horner(x, coeffs):
    n = len(coeffs) - 1
    p = coeffs[n] # Start with highest degree coefficient (an)
    
    for i in range(n - 1, -1, -1):
        p = p * x + coeffs[i]
    return p
    

def interpolation_graph(year, gdb):
    coeffs = monomial(year, gdb)
    
    gdp_1992 = horner(1992, coeffs)
    print(f"Estimated GDP in 1992: {gdp_1992:.6f}")

    
    plt.scatter(year, gdb, color='red', label='Data Points')
    
    x_range = np.linspace(min(year), max(year), 100)
    y_interp = [horner(t, coeffs) for t in x_range]
    
    plt.plot(x_range, y_interp, color='blue', label='Monomial Interpolation')
    plt.axvline(x=1992, color='green', linestyle='--', label='1992 Prediction')
    
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.title('GDP Interpolation using Vandermonde Matrix')
    plt.legend()
    plt.show()

interpolation_graph(year, gdb)

