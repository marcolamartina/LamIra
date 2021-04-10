import cv2
import os
import sys
from math import copysign, log10
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import random
import math
import itertools

dir_name=os.path.dirname(__file__)


def hu_log(humoments):
    values=[]
    for i in range(0,len(humoments)):
        # Log transform Hu Moments to make
        # squash the range
        if humoments[i][0]==0:
            values.append(40)
        else:        
            values.append(-1*copysign(1.0,humoments[i][0])*log10(abs(humoments[i][0])))
    ref_log=np.array(values)
    return ref_log  


def distance_image(im1,im2):
    return cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)

def moments_distance(hu1,hu2):
    manhatten=np.sum(np.abs(hu1[:3]-hu2[:3]))
    euclidean=np.linalg.norm(hu1[:3]-hu2[:3])
    return manhatten

def get_roi(image,tollerance=5):
    min_x,min_y,w,h = cv2.boundingRect(image)
    max_x=min_x+w
    max_y=min_y+h
    min_x=max(0,min_x-tollerance)
    min_y=max(0,min_y-tollerance)
    max_x=min(640,max_x+tollerance)
    max_y=min(480,max_y+tollerance)
    start=(min_x,min_y)
    end=(max_x,max_y)
    result = image[start[1]:end[1], start[0]:end[0]]
    
    percentage=max([(0,result.shape[0]/480),(1,result.shape[1]/640)],key=lambda x:x[1])
    if percentage[0]==0:
        result=cv2.resize(result,(int(result.shape[1]/percentage[1]),480),cv2.INTER_AREA)
    else:
        result=cv2.resize(result,(640,int(result.shape[0]/percentage[1])),cv2.INTER_AREA)    

    return result           

def humoments(filename):
    path=os.path.join(dir_name,filename)
    # Read image
    im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #im = ~im

    # Threshold image
    _,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    im=get_roi(im)


    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    gradient = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, element)

    # Calculate Moments
    moment = cv2.moments(gradient)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moment)

    return im,huMoments

def main():
    files = os.listdir( dir_name )
    files=[i for i in files if i.endswith(".png") and (i[:3]=="tri" or i[:3]=="thu" or i[:3]=="fis" or "byhand" in i)]
    files.sort()
    ref=files[0]
    ref_im,ref_humoments=humoments(ref)
    ref_log=hu_log(ref_humoments)
    humoments_list={}
    for filename in files:
        im,curr_humoments=humoments(filename)
        curr_humoments_log=hu_log(curr_humoments)
        # Print Hu Moments
        print("Log transform Hu moments {}: ".format(filename),end='')
        for i in range(0,7):     
            print("{:.4f}".format(curr_humoments_log[i]),end=' ')
        print("\n")    
        shape=filename.split('-')[0]
        if shape in humoments_list.keys():   
            humoments_list[shape].append(curr_humoments_log[:3])
        else:
            humoments_list[shape]=[curr_humoments_log[:3]]        

    hulls=[(ConvexHull(pts),shape_label) for shape_label,pts in humoments_list.items()]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1, projection="3d")
    colors=[(1, 0, 0),(0, 0, 0),(0, 1, 0),(0, 0, 1),(0, 1, 1),(1, 0, 1),(1, 1, 0)]

    ax.set_title('Shape')
    p=np.array([1.52,5.0,9.0])
    ax.plot(p.T[0], p.T[1], p.T[2], "o", color=(0,1,0))
    for i,hull_and_label in enumerate(hulls):
        hull,shape_label=hull_and_label
        distance=distance_hull_point(p,hull)
        print(shape_label,distance) 
        for s,plane in zip(hull.simplices,hull.equations):
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(hull.points.T[0], hull.points.T[1], hull.points.T[2], "o", color=colors[i])
            ax.plot(hull.points[s, 0], hull.points[s, 1], hull.points[s, 2],"-", color=colors[i])
            ax.text(hull.points[0][0],hull.points[0][1],hull.points[0][2], shape_label, size=10, zorder=1, color='k')
        #ax.annotate(str(i), (hull.points[0]))

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    plt.show()


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

        

if __name__ == "__main__":
    main()