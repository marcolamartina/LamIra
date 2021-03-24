import numpy as np
import cv2
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from collections import Counter

def euclidean_distance(a,b):
    xa,ya,za=a
    xb,yb,zb=b
    return math.sqrt((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)

def line_from_2_points(p1,p2):
    xa,ya=p1
    xb,yb=p2
    a=ya-yb
    b=xb-xa
    c=xa*(yb-ya)-ya*(xb-xa)
    return a,b,c

def distance_point_line(p,line):
    a,b,c=line
    xp,yp=p
    return abs(a*xp+b*yp+c)/math.sqrt(a**2+b**2)


def dbscan(points, distance_measure=euclidean_distance, min_samples_frac=100, eps=2.9, dimensions=False):
    length=len(points)
    if not length:
        return []
    min_samples=int(length/min_samples_frac)   
    scaler = StandardScaler().fit(points)
    X = scaler.inverse_transform(scaler.transform(points))
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_measure).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    clustered_centers_x=[]
    clustered_centers_y=[]
    clustered_centers_z=[]
    for k in range(n_clusters_):
        my_members = labels == k
        clustered_centers_x.append(np.median(X[my_members,0]))
        clustered_centers_y.append(np.median(X[my_members,1]))
        clustered_centers_z.append(np.median(X[my_members,2]))
    output=[]
    for point in zip(clustered_centers_x,clustered_centers_y,clustered_centers_z):
        output.append(point)
    if dimensions:
        c=Counter(labels)            
        return output,[c[i] for i in c.keys() if i >=0]
    else:    
        return output  

def kmeans(points, n_clusters=None, dimensions=True):
    algorithm_type='k-means++'
    if n_clusters==None:
        max_n_clusters=6

        elbow_points=[]
        for i in [1, max_n_clusters]:
                kmeans = KMeans(n_clusters = i, init = algorithm_type, random_state = 0)
                kmeans.fit(points)
                elbow_points.append((i,kmeans.inertia_))
        line=line_from_2_points(*elbow_points)
        max_distance=0
        n_clusters=0
        for i in range(2,max_n_clusters):
            kmeans = KMeans(n_clusters = i, init = algorithm_type, random_state = 0)
            kmeans.fit(points)
            p=(i,kmeans.inertia_)
            distance=distance_point_line(p,line)
            if distance<=max_distance:
                break
            n_clusters=i
            max_distance=distance            

    kmeans = KMeans(n_clusters=n_clusters, init =algorithm_type, max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(points)
    labels={i:0 for i in range(n_clusters)}
    for l in kmeans.labels_:
        labels[l]+=1
    centroids=[(v.tolist(),labels[i]/len(points)) for i,v in enumerate(kmeans.cluster_centers_)]
    centroids.sort(key=lambda x:x[1],reverse=True)
    if not dimensions:
        centroids=[c[0] for c in centroids]
    return centroids


