import numpy as np
import matplotlib.pyplot as plt

next = ""

# Import data
xy =  np.loadtxt('./test.csv',delimiter=',',dtype=np.int64)
X = xy[:,1:]
y = xy[:,0]
print(X.shape[1])

# See plots
i = 0
while next == "":
    xs = [x/204000.0 for x in range(1,X.shape[1])]
    ys = X[i,1:]
    plt.plot(xs,ys)

    # Tomar rango de -50000 a 50000 como silencio
    plt.axhline(y = 50000, color = 'r', linestyle = '-', linewidth=1) 
    plt.axhline(y = -50000, color = 'r', linestyle = '-', linewidth=1) 
    plt.show()

    next = input("[Enter] to continue [s] to stop: ")
    i += 1
    plt.close()