import numpy as np
import matplotlib.pyplot as plt

years , gdb = np.loadtxt(r"G:\Other computers\Sase-Um6p\Documents\Sase-Docs\Python-Comp\Comp-Workspace-\interpolation\data.txt", unpack=True)
years , gdb = np.loadtxt(r"G:\Other computers\Sase-Um6p\Documents\Sase-Docs\Python-Comp\Comp-Workspace-\interpolation\data2.txt", unpack=True)



def lagrange_interpolation(x, years, gdp):
    n = len(years)-1
    p = 0
    for i in range(n+1):
        l = 1
        for k in range(n+1) :
            if i != k : 
                l = l * (x - years[k]) / (years[i] - years[k])
            
        p = p + gdp[i] * l

    return p
    
result = lagrange_interpolation(1992, years, gdb)

print(f"Interpolated GDP for the year 1992: {result}")

def interpolation_graph(years, gdb): 
    graph = np.stack((years, gdb), axis=1)
    interpolation_graph = np.vectorize(lambda x: lagrange_interpolation(x, years, gdb))
    plt.scatter(graph[:,0], graph[:,1], color='red', label='Data Points')
    x = np.linspace(min(years), max(years), 100)
    plt.plot(x, interpolation_graph(x), color='blue', label='Lagrange Interpolation')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.legend()
    return plt.show()

interpolation_graph(years, gdb)






