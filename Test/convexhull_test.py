import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import random
import math
import itertools

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def get_corners(vertices):
    return list(itertools.combinations(vertices,2))

def distance_hull_point(point, hull):
    if point_in_hull(point, hull):
        return 0.0
    return min([distance_point_face(point,(hull.points[s[0]],s[1])) for s in zip(hull.simplices,hull.equations)])    

def project(point, plane, tolerance=1e-12):
    if abs(np.dot(plane[:-1], point) + plane[-1]) <= tolerance:
        return point
    t = -(plane[-1] + np.dot(plane[:-1], point))/(np.sum(plane[:-1]**2))
    return point + plane[:-1]*t  

def triangle_area(vertices):
    segments=list(itertools.combinations(vertices,2))
    segments_len=[np.linalg.norm(s[0]-s[1]) for s in segments]
    semiperimeter=sum(segments_len)/2
    result=semiperimeter
    for l in segments_len:
        result*=semiperimeter-l
    return math.sqrt(result)
   
def in_triangle(point_projection,vertices,corners=None,tolerance=1e-12):
    if not corners:
        corners=get_corners(vertices)
    total_area=triangle_area(vertices)
    sub_triangle_areas=[(triangle_area([point_projection,p1,p2])/total_area) for p1,p2 in corners]
    return all(0<=t<=1 for t in sub_triangle_areas) and 1-tolerance<=sum(sub_triangle_areas)<=1+tolerance

def distance_point_corner(p,corner):
    a, b = corner
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

def distance_point_plane(point,plane):
    return abs(plane[-1] + np.dot(plane[:-1], point))/math.sqrt(np.sum(plane[:-1]**2))

def distance_point_face(point,face):
    vertices,plane=face
    point_projection=project(point,plane)
    corners=get_corners(vertices)
    if in_triangle(point_projection,vertices,corners):
        return distance_point_plane(point,plane)
    else:
        return min([distance_point_corner(point,c) for c in corners])     
      


pts=[[[random.randint(0,100),random.randint(-128,128),random.randint(-128,128)] for _ in range(4)]]
pts=[[[random.randint(0,100) for _ in range(3)] for _ in range(10)]]
rng = np.random.RandomState(0)
pts = rng.random_sample((40, 30))
#pts=[[[0,0,0.00001],[0.00001,0,0.00001],[0.00001,0,0.00001],[0,0.00001,0],[0.00001,0,0],[0.00001,0,0.00001],[0.00001,0,0.00001],[0.00001,0,0.00001]]]
#pts=[np.array(i) for i in pts]
hulls=ConvexHull(pts)
print(hulls)
exit(0)
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1, projection="3d")
ax2 = fig.add_subplot(1,3,2, projection="3d")
ax3 = fig.add_subplot(1,3,3, projection="3d")
axes=[ax1,ax2,ax3]
colors=[(0.1, 0.2, 0.5),(0, 0, 0),(0.1, 0.2, 0.5),(0, 0, 0),(0.1, 0.2, 0.5)]


# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
for ax in axes:
    ax.set_title('subplot 1')
    p=np.array([random.randint(0,100),random.randint(-128,128),random.randint(-128,128)])
    p=np.array([60,0,0])
    p=np.array([random.randint(0,100) for _ in range(3)])
    ax.plot(p.T[0], p.T[1], p.T[2], "o", color=(0,1,0))
    for i,hull in enumerate(hulls):
        distance=distance_hull_point(p,hull)
        print(distance) 
        for s,plane in zip(hull.simplices,hull.equations):
            ax.plot(p.T[0], p.T[1], p.T[2], "o", color=(1,0,0))
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(hull.points.T[0], hull.points.T[1], hull.points.T[2], "o", color=colors[i])
            ax.plot(hull.points[s, 0], hull.points[s, 1], hull.points[s, 2],"-", color=colors[i])
            ax.text(hull.points[0][0],hull.points[0][1],hull.points[0][2], "blu", size=10, zorder=1, color='k')
        #ax.annotate(str(i), (hull.points[0]))



# Make axis label
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))

plt.show()


