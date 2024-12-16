import numpy as np
import matplotlib.pyplot as plt

xy =  np.loadtxt('./result.csv',delimiter=',',dtype=np.int64)
X = xy[:,1:]
y = xy[:,0]
xs = [x/204000.0 for x in range(1,xy.size)]
#ys = xy[1:]
#plt.plot(xs,ys)
#plt.show()
print(xy.shape)