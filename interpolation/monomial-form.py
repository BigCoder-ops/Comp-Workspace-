import numpy as np
import matplotlib.pyplot as plt


year , gdb = np.loadtxt("data.txt", unpack=True)

def monomial(year, gdb):
    n = len(year) - 1
    v  = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    for i in range(n+1):
        for j in range(n+1):
            v[i,j] = year[i]**j
            
    for i in range(n+1):
        b[i] = gdb[i]

    solve = np.linalg.solve(v, gdb)
    return solve

def interpolation_graph(year, gdb):
    graph = np.stack((year, gdb), axis=1)

    interpolation_graph = np.poly1d(monomial(year, gdb))
    plt.scatter(graph[:,0], graph[:,1], color='red', label='Data Points')
    x = np.linspace(min(year), max(year), 100)
    plt.plot(x, interpolation_graph(x), color='blue', label='Monomial Interpolation')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.legend()
    return plt.show()

monomial(year, gdb) 
interpolation_graph(year, gdb)
