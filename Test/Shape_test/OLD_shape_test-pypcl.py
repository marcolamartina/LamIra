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
from sklearn.mixture import GaussianMixture
import pclpy
from pclpy import pcl
from sklearn.neighbors import KDTree

SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF= 0.2    # scaling factor for difference in normals wrt. difference in position for points,
                                                # when computing difference between two point clouds. 
                                                # 0 => Only look at difference between point and its closest point
                                                # Higher value => matching normals are more important than point distances
SMOOTHNESS_MEASURE_NUMBINS = 8                  # Number of bins in histogram. We found 8 to work best quite consistently.
NNRADIUS = 0.004                                # Used in Local Convexity and Smoothness measure for local neighborhood finding

dir_name=os.path.dirname(__file__)
image_path = os.path.join(dir_name, "object")

def get_image(filename):
    path=os.path.join(image_path,filename)
    if "depthcrop" in filename or 'maskcrop' in filename:
        im = cv2.imread(path,0)
    else: 
        im = cv2.imread(path)   
    return im

def apply_mask(mask,image):
    i=image.copy()
    i[mask == 0]=0
    return i

def depth_to_meter(depth):
    depth=depth.astype(float)
    try:
        return 1/((depth * 4 * -0.0030711016) + 3.3309495161)
    except:
        return 0.0    

# just return mean of distances from points in cloud1 to their nearest neighbors in cloud2 
def cloudAlignmentScoreDense(cloud1, cloud2):
    tree = KDTree(cloud2)
    N=cloud1.shape[0]
    accum=0.0
    result = tree.query(cloud1, k=1)
    for i,(dist, ind) in enumerate(zip(*result)):
        accum += dist[0]
    return accum/N

def cloudAlignmentScoreDenseWithNormalsNormalized(cloud1, normals1, cloud2, normals2, relweight, dnormalize):
    tree = KDTree(cloud2)
    N=cloud1.shape[0]
    accum=0.0
    result = tree.query(cloud1, k=1)
    for i,(dist, ind) in enumerate(zip(*result)):
        accum += dist[0] / dnormalize
        dot = np.dot(normals1[i],normals2[ind[0]])
        accum += relweight*(1.0 - dot)   
    return accum/N

def calculate_compactness_3d(points):    
    max_length = np.max(points,axis=0)[0]
    min_length = np.min(points,axis=0)[0]
    return points.shape[0] / (max(max_length-min_length, 0.0000001)**2)

def calculate_symmetry_3d(points_np, normals, relweight=SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF):
    mins=points_np.min(axis=0)
    maxes=points_np.max(axis=0)
    ranges = maxes - mins
    ranges /= ranges.sum()
    score=0.0
    for i,vector in enumerate(np.array([[-1,1,1],[1,-1,1],[1,1,-1]])):
        dest=points_np*vector
        normdest=normals*vector
        overlap = cloudAlignmentScoreDenseWithNormalsNormalized(points_np, normals, dest, normdest, relweight, ranges[i])\
                 +cloudAlignmentScoreDenseWithNormalsNormalized(dest, normdest, points_np, normals, relweight, ranges[i])    
        score += ranges[i]*overlap
    return -score


def calculate_global_convexity_3d(points):
    hull=ConvexHull(points)
    overlap= cloudAlignmentScoreDense(points, hull.points[hull.vertices])
    return -overlap

def calculate_local_convexity_and_smoothness_3d(points, normals, NNradius=NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
    tree = KDTree(points)
    N=points.shape[0]
    score=0.0
    Hs=0.0
    bins=np.ones(NUMBINS)
    neighbors = tree.query_radius(points, NNradius)
    for i,(p1,n1,neighbors_current) in enumerate(zip(points,normals,neighbors)):
        binsum = NUMBINS
        n2=(np.random.rand(3)+1)/2
        n2=n2-np.dot(n1,n2)*n1
        d = np.linalg.norm(n2)
        n2 /= d
        n3 = np.cross(n1,n2)
        dot=0.0
        nc=0
        for j in neighbors_current:
            if j==i:
                continue    
            v = p1-points[j]
            d = np.linalg.norm(v)
            v/=d
            dot = np.dot(n1,v)
            if dot > 0.0:
                nc += 1
            dot1 = np.dot(n2,v)/d
            dot2 = np.dot(n3,v)/d
            theta = ((np.arctan2(dot1, dot2)+np.pi)/2)/np.pi # angle in range 0->1
            binid = int((theta-0.001)*NUMBINS)
            bins[binid] += 1
            binsum+=1             
        score += (1.0*nc)/len(neighbors_current)
        bins/=binsum
        H=-(bins*np.log(bins)).sum()
        if not np.isnan(H):
            Hs += H       
        
    return score/N,Hs/N  

def calculate_local_convexity_3d(points, normals, NNradius=NNRADIUS):
    tree = KDTree(points)
    N=points.shape[0]
    score=0.0
    neighbors = tree.query_radius(points, NNradius)
    for i,(p,normal,neighbors_current) in enumerate(zip(points,normals,neighbors)):
        dot=0.0
        nc=0
        for j in neighbors_current:
            if j==i:
                continue    
            v = p-points[j]
            d = np.linalg.norm(v)
            v/=d
            dot = np.dot(normal,v)
            if dot > 0.0:
                nc += 1 
                     
        score += (1.0*nc)/len(neighbors_current)
    return score/N

def calculate_smoothness_3d(points, normals, NNradius=NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
    Hs=0.0
    tree = KDTree(points)
    N=points.shape[0]
    bins=np.ones(NUMBINS)
    neighbors = tree.query_radius(points, NNradius)
    for i, (p1,n1,neighbors_current) in enumerate(zip(points,normals,neighbors)):
        #print("{:.2f}%".format(i*100/len(points)))
        binsum = NUMBINS
        n2=(np.random.rand(3)+1)/2
        dot=np.dot(n1,n2)
        n2=n2-dot*n1
        d = np.linalg.norm(n2)
        n2 /= d
        n3 = np.cross(n1,n2)
        for j in neighbors_current:
            if j==i:
                continue
            p2=points[j]
            v = p1-p2
            d = np.linalg.norm(v)
            v/=d
            dot1 = np.dot(n2,v)/d
            dot2 = np.dot(n3,v)/d
            theta = ((np.arctan2(dot1, dot2)+np.pi)/2)/np.pi # angle in range 0->1
            binid = int((theta-0.001)*NUMBINS)
            bins[binid] += 1
            binsum+=1
        bins/=binsum
        H=-(bins*np.log(bins)).sum()
        if not np.isnan(H):
            Hs += H
    return Hs/N # high entropy = good.

def pad_image(depth,result_shape=(480,640)):
    top=int((result_shape[0]-depth.shape[0])/2)
    bottom=result_shape[0]-top-depth.shape[0]
    left=int((result_shape[1]-depth.shape[1])/2)
    right=result_shape[1]-left-depth.shape[1]
    return np.pad(depth, ((top, bottom), (left, right)), 'constant')

def depth_to_cloud(depth_original):
    depth=depth_to_meter(depth_original)
    depth=pad_image(depth)
    depth_original=pad_image(depth_original)
    cameraMatrix = np.array(
        [[525., 0., 320.0],
         [0., 525., 240.0],
         [0., 0., 1.]])
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]

    xyz_offset=[0,0,depth.min()]
    array = []
    xy=np.argwhere((depth_original<255) & (depth_original>0)) 
    xy=xy.astype(float)
    z=depth[np.where((depth_original<255) & (depth_original>0))]
    z=z.astype(float)  
    a=((xy[:,0]-ox)*z*inv_fx)
    b=((xy[:,1]-oy)*z*inv_fy)
    xy[:,0]=b
    xy[:,1]=a
    xyz=np.insert(xy, 2, values=z, axis=1)     
    return pcl.PointCloud.PointXYZ.from_array(xyz)

def get_cloud_and_normals(depth):
    point_cloud = depth_to_cloud(depth)

    # compute mls
    mls = point_cloud.moving_least_squares(search_radius=0.05, compute_normals=True, num_threads=8)
    
    # get points and normals
    normals = mls.normals
    cloud = point_cloud.xyz

    # removing Nan
    cloud=cloud[~np.isnan(cloud).any(axis=1)]
    normals=normals[~np.isnan(normals).any(axis=1)]

    return cloud,normals

def calculate_descriptor(mask, rgb, depth):
    descriptors_2d=calculate_descriptors_2d(mask,depth)
    descriptors_3d=calculate_descriptors_3d(mask,rgb,depth)
    return descriptors_2d+descriptors_3d

def calculate_descriptors_3d(mask,rgb,depth):
    cloud,normals=get_cloud_and_normals(depth)

    compactness_3d=calculate_compactness_3d(cloud)
    symmetry_3d=calculate_symmetry_3d(cloud, normals)
    global_convexity_3d=calculate_global_convexity_3d(cloud)
    local_convexity_3d,smoothness_3d=calculate_local_convexity_and_smoothness_3d(cloud, normals)

    return [compactness_3d,symmetry_3d,global_convexity_3d,local_convexity_3d,smoothness_3d]

def get_roi(image):
    min_x,min_y,w,h = cv2.boundingRect(image)
    max_x=min_x+w
    max_y=min_y+h
    return image[min_y:max_y,min_x:max_x]

def calculate_descriptors_2d(mask,rgb):
    mask_roi=get_roi(mask)
    compactess_2d=calculate_compactess_2d(mask_roi)
    symmetry_2d=calculate_symmetry_2d(mask_roi)
    global_convexity_2d=calculate_global_convexity_2d(mask)
    histogram, uniqueness_2d=calculate_uniqueness_2d(mask,rgb)
    smoothness_2d=calculate_smoothness_2d(mask,rgb,histogram)
    return [compactess_2d,symmetry_2d,global_convexity_2d,uniqueness_2d,smoothness_2d]

def calculate_compactess_2d(mask):
    pixels_on = cv2.countNonZero(mask)
    pixels = mask.shape[0] * mask.shape[1]
    return pixels_on/pixels    



def calculate_symmetry_2d(mask):
    symmetries=[]
    for i in range(2):
        if i:
            mask=mask.T
        half=int(mask.shape[1]/2)
        first_half = mask[:, 0:half]
        second_half = mask[:, half+(mask.shape[1] % 2):]
        second_half = np.flip(second_half, axis=1)
        symmetry = np.sum(first_half == second_half)
        symmetries.append(symmetry/first_half.size)
        
    return max(symmetries)

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def calculate_global_convexity_2d(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(contours[0])
    m=mask.copy()
    cv2.drawContours(m, [hull], -1, 150, 1)
    contours_pos=np.argwhere(m==150)
    points=np.argwhere(m==255)
    result=np.average(np.min(np.linalg.norm(contours_pos - points[:,None], axis=-1),axis=1))
    return result

def get_angle(v1,v2):
    angles=np.array([[135,120,90,60,45],
                    [150,135,90,45,30],
                    [180,180,0,0,0],
                    [210,225,270,315,330],
                    [225,240,270,300,315]])
    return (angles[v1[0],v1[1]]-angles[v2[0],v2[1]])%180                    

def entropy(hist):
    return -sum([i*math.log(i) for i in hist])

def calculate_uniqueness_2d(mask,rgb,show_hist=False):
    hist={}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    m=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(m, contours, -1, (0,255,255), 1)
    t=3
    contours=contours[0]
    l=len(contours)
    vectors=[contours[(i+2)%l]-contours[(i-2)%l] for i in range(0,l,t)]
    l=len(vectors)
    for i in range(l):
        angle=get_angle(vectors[i][0],vectors[(i+1)%l][0])
        if angle in hist.keys():
            hist[angle]+=1
        else:
            hist[angle]=1
    if show_hist:
        from collections import Counter
        num = Counter(hist)
        x = []
        y = []
        for k in sorted(hist.keys()):
            x.append(hist[k])
            y.append(k)

        x_coordinates = np.arange(len(num.keys()))
        plt.bar(x_coordinates,x)
        plt.xticks(x_coordinates,y)
        plt.show()          
    h=[i/l for i in hist.values()]  
    h2=[(k,v) for k,v in hist.items()]        
    return h2,entropy(h)

def calculate_smoothness_2d(mask,rgb,histogram):
    X = np.array(histogram)
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    return np.max(gm.means_[:,0])


def main():
    files = os.listdir( image_path )
    files.sort()
    crop_rgb=[get_image(i) for i in files if i.endswith("_crop.png")]
    crop_masks=[get_image(i) for i in files if i.endswith("maskcrop.png")]
    crop_depth=[get_image(i) for i in files if i.endswith("depthcrop.png")]
    names=[" ".join(i.split("_")[0:-4]) for i in files if i.endswith("depthcrop.png")]
    
    depths = [ apply_mask(m,i) for m, i in zip(crop_masks,crop_depth)]
    rgbs = [ apply_mask(m,i) for m, i in zip(crop_masks,crop_rgb)]

    # cv2.imshow("Both", np.hstack((crop_rgb[1], crop_masks[1])))
    # cv2.waitKey(0)

    ref=files[0]
    descriptors_dict={}
    for name,mask,rgb,depth in zip(names,crop_masks,rgbs,depths):
        descriptors=calculate_descriptor(mask,rgb,depth)
        print("Descriptors {}: ".format(name),end='')
        for d in descriptors:    
            print("{:.4f}".format(d),end=' ')
        print("\n")    
        if name in descriptors_dict.keys():   
            descriptors_dict[name].append(descriptors)
        else:
            descriptors_dict[name]=[descriptors]        

        


def distance_point_point(p1,p2):
    return np.linalg.norm(p1-p2)

def distance_hull_external_poins_to_point(point, hull):
    if point_in_hull(point, hull):
        return 0.0
    return min([distance_point_point(point,v) for v in hull.vertices])

        

if __name__ == "__main__":
    main()