import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import random
import time 

rng = np.random.RandomState(0)
possible_dimensions=range(5,100,5)
possible_dimensions_sample=range(200,10001,200)

dimension=9
x_sample=[]
y_sample=[]
for dimension_sample in possible_dimensions_sample:
    pts = rng.random_sample((dimension_sample, dimension))
    starting=time.time()
    tree = KDTree(pts)
    time_elapsed=time.time()-starting
    x_sample.append(dimension_sample)
    y_sample.append(time_elapsed*1000)
    print(time_elapsed)

dimension_sample=1000
x_dim=[]
y_dim=[]
for dimension in possible_dimensions:
    pts = rng.random_sample((dimension_sample, dimension))
    starting=time.time()
    tree = KDTree(pts)
    time_elapsed=time.time()-starting
    x_dim.append(dimension)
    y_dim.append(time_elapsed*1000)
    print(time_elapsed)

plt.subplot(2, 1, 1)
plt.title("KDTree Benchmark")
plt.xlabel("Sample Dimension")
plt.ylabel("Time(ms)")
plt.plot(x_sample,y_sample)

plt.subplot(2, 1, 2)
plt.xlabel("Dimensionality")
plt.ylabel("Time(ms)")
plt.plot(x_dim,y_dim)

plt.show()    
