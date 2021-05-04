import numpy as np
import random
from sklearn.neighbors import KDTree


rng = np.random.RandomState(0)
X = rng.random_sample((1000, 10))  # 10 points in 3 dimensions
tree = KDTree(X,metric='euclidean')        
#print(X)
p=rng.random_sample((1, 10))
point=np.array(p[0])                
dist, ind = tree.query(p, k=100)
f=tree.get_arrays()
real_distances=[]
distances=[]

# Get points from index
#print(np.asarray(tree.data[index]))


for distance,index in zip(dist[0],ind[0]): 
    print("indice {} punto {}".format(index,f[0][index]))  # indices of 3 closest neighbors
    print("distanza: {}".format(distance))  # distances to 3 closest neighbors

