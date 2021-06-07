import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
import math
import time

rng = np.random.RandomState(0)
possible_dimensions=range(2,14,1)
possible_dimensions_sample=range(20,201,20)

dimension=9
x_sample=[]
y_sample=[]
for dimension_sample in possible_dimensions_sample:
    pts = rng.random_sample((dimension_sample, dimension))
    starting=time.time()
    hull=ConvexHull(pts)
    time_elapsed=time.time()-starting
    x_sample.append(dimension_sample)
    y_sample.append(time_elapsed)
    print(time_elapsed)

dimension_sample=60
x_dim=[]
y_dim=[]
for dimension in possible_dimensions:
    pts = rng.random_sample((dimension_sample, dimension))
    starting=time.time()
    hull=ConvexHull(pts)
    time_elapsed=time.time()-starting
    x_dim.append(dimension)
    y_dim.append(time_elapsed)
    print(time_elapsed)

plt.subplot(2, 1, 1)
plt.title("Convex Hull Benchmark")
plt.xlabel("Sample Dimension")
plt.ylabel("Time(s)")
plt.plot(x_sample,y_sample)

plt.subplot(2, 1, 2)
plt.xlabel("Dimensionality")
plt.ylabel("Time(s)")
plt.plot(x_dim,y_dim)

plt.show()    