import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
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

output_file=__file__[:-3]+".pdf"
filename=__file__.split("/")[-1][:-3]
title=" ".join(filename.split("_"))

ax=plt.subplot(2, 1, 1)
#plt.title(title)
plt.xlabel("Samples")
plt.ylabel("Time(s)")
plt.plot(x_sample,y_sample, color=[0,0,0], linewidth=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax=plt.subplot(2, 1, 2)
plt.xlabel("Dimensionality")
plt.ylabel("Time(s)")
plt.plot(x_dim,y_dim, color=[0,0,0], linewidth=0.8)
plt.tight_layout()


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.savefig(output_file, bbox_inches="tight")
plt.savefig("/Users/marco/Desktop/{}.pdf".format(filename), bbox_inches="tight")
#plt.show()   
