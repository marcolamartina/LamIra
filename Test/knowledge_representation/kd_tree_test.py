import numpy as np
import random
from sklearn.neighbors import KDTree
import pickle
import os

dim=2
rng = np.random.RandomState(0)
X = rng.random_sample((100, dim))  # 10 points in 3 dimensions
tree = KDTree(X)        
#print(X)
p=rng.random_sample((1, dim))
point=np.array(p[0])                
dist, ind = tree.query(p, k=1)
f=tree.get_arrays()
real_distances=[]
distances=[]

        
s = pickle.dumps(tree)
with open(os.path.dirname(__file__)+"/pippo.pickle","wb") as fl:
    fl.write(s)                     
with open(os.path.dirname(__file__)+"/pippo.pickle","rb") as fl2:
    tree=pickle.loads(fl2.read()) 


points=np.array(tree.get_arrays()[0])
mins=points.min(axis=0)
maxes=points.max(axis=0)
centroid=points.mean(axis=0)
ellipsoid = (maxes - mins)/2

def in_ellipsoid(point, ellipsoid, centroid):
    if 0 in ellipsoid:
        return False
    return (np.square(point-centroid)/np.square(ellipsoid)).sum()<1

p=np.array([3.9,0])
ellipsoid=np.array([4,6])
centroid=np.array([0,0])

print(in_ellipsoid(p, ellipsoid, centroid))

# Get points from index
#print(np.asarray(tree.data[index]))


for distance,index in zip(dist[0],ind[0]): 
    print("indice {} punto {}".format(index,f[0][index]))  # indices of 3 closest neighbors
    print("distanza: {}".format(distance))  # distances to 3 closest neighbors

