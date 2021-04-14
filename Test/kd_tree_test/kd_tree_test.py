import numpy as np
import random
from sklearn.neighbors import KDTree


rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
X=[random.randint(10) for _ in range(10)]
tree = KDTree(X, leaf_size=2)        
                
dist, ind = tree.query(X[:1], k=3)     
print(ind)  # indices of 3 closest neighbors

print(dist)  # distances to 3 closest neighbors
