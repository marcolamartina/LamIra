#import laspy
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import itertools
import pclpy
from pclpy import pcl


SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF= 0.2    # scaling factor for difference in normals wrt. difference in position for points,
                                                # when computing difference between two point clouds. 
                                                # 0 => Only look at difference between point and its closest point
                                                # Higher value => matching normals are more important than point distances
LOCAL_CONVX_MEASURE_NNRADIUS = 0.0035           # (before 0.0075) used in Local Convexity measure for local neighborhood
SMOOTHNESS_MEASURE_NNRADIUS = 0.005             # used in Smoothness measure for local neighborhood finding
SMOOTHNESS_MEASURE_NUMBINS = 8                  # Number of bins in histogram. We found 8 to work best quite consistently.
NNRADIUS = 0.004

develop_mode=False

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

def score_compactness(points):
    if develop_mode:
        return 0    
    max_length = np.max(points,axis=0)[0]
    min_length = np.min(points,axis=0)[0]
    return points.shape[0] / (max(max_length-min_length, 0.0000001)**2)

def score_symmetry(points_np, normals, relweight=SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF):
    if develop_mode:
        return 0
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


def score_global_convexity(points):
    if develop_mode:
        return 0
    hull=ConvexHull(points)
    overlap= cloudAlignmentScoreDense(points, hull.points[hull.vertices])
    return -overlap

def score_local_convexity_and_smoothness(points, normals, NNradius=NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
    if develop_mode:
        return 0
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
  

def score_local_convexity(points, normals, NNradius=LOCAL_CONVX_MEASURE_NNRADIUS):
    if develop_mode:
        return 0
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

def score_smoothness(points, normals, NNradius=SMOOTHNESS_MEASURE_NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
    if develop_mode:
        pass
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


def depth_to_meter(depth):
    depth=depth.astype(float)
    try:
        return 1/((depth * 4 * -0.0030711016) + 3.3309495161)
    except:
        return 0.0    

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

def main():
    path = os.path.dirname(__file__)
    #depth0 = cv2.imread(path+'/test_data/depth.png', 0)
    files = os.listdir('/home/davide/LamIra/Test/segmentation_test/test/')
    for f in files:
        if f.endswith('.png'):
            depth0 = cv2.imread('/home/davide/LamIra/Test/segmentation_test/test/'+f, 0)
            print(f)
            #depth0 = cv2.resize(depth0, (round(width / 2), round(height / 2)), interpolation=cv2.INTER_AREA)
            point_cloud = depth_to_cloud(depth0)

            # compute mls
            mls = point_cloud.moving_least_squares(search_radius=0.05, compute_normals=True, num_threads=8)
            
            # get points and normals
            normals = mls.normals
            cloud = point_cloud.xyz

            # removing Nan
            cloud=cloud[~np.isnan(cloud).any(axis=1)]
            normals=normals[~np.isnan(normals).any(axis=1)]

            compactness=score_compactness(cloud)
            symmetry=score_symmetry(cloud, normals)
            global_convexity=score_global_convexity(cloud)
            local_convexity,smoothness=score_local_convexity_and_smoothness(cloud, normals)
            print("Compactness: {}\nSymmetry: {}\nGlobal Convexity: {}\nLocal Convexity: {}\nSmoothness: {}".format(compactness, symmetry, global_convexity, local_convexity, smoothness))
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    params = mls.x,mls.y,mls.z,mls.normal_x,mls.normal_y,mls.normal_z
    r=[p[~np.isnan(p)] for p in params]
    r_c=[]
    for channel in r:
        r_c.append(np.array([elem for i,elem in enumerate(channel) if i%100==0]))   
    X, Y, Z, U, V, W=r_c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W,length=0.1)
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])
    ax.set_zlim([np.min(Z), np.max(Z)])
    plt.show()
    '''

if __name__=="__main__":
    main()

