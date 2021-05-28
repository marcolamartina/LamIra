import cv2
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import random
import math
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
try:
    import pcl
    dummy=False
except:
    dummy=True    

SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF= 0.2    # scaling factor for difference in normals wrt. difference in position for points,
                                                # when computing difference between two point clouds. 
                                                # 0 => Only look at difference between point and its closest point
                                                # Higher value => matching normals are more important than point distances
SMOOTHNESS_MEASURE_NUMBINS = 8                  # Number of bins in histogram. We found 8 to work best quite consistently.
NNRADIUS = 0.004                                # Used in Local Convexity and Smoothness measure for local neighborhood finding


try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_shape_extractor = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_shape_extractor = os.path.dirname(__file__)
    data_dir_images = os.path.join(data_dir_shape_extractor,"..","..","Datasets","rgbd-dataset")
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))


class Shape_extractor:
    def extract(self,image):
        if dummy:
            return [random.random() for _ in range(10)]
        descriptors=self.calculate_descriptors(image)
        return descriptors

    def extract_humoments(self,image):
        humoments=self.calculate_humoments(image)
        humoments_log_trasformed=self.hu_log_trasform(humoments)
        return humoments_log_trasformed

    def classify(self,features):
        labels=["tondo", "quadrato", "esagono", "triangolo", "rettangolo", "ovale"]
        return [(l,random.random()) for l in labels]    

    def hu_log_trasform(self,humoments,default=40):
        values=[]
        for i in range(0,len(humoments)):
            if humoments[i][0]==0:
                values.append(default)
            else:        
                values.append(-1*math.copysign(1.0,humoments[i][0])*math.log10(abs(humoments[i][0])))
        ref_log=np.array(values)
        return ref_log  

    def moments_distance(self,hu1,hu2):
        return np.linalg.norm(hu1[:3]-hu2[:3])

    def get_roi(self,image,tollerance=5):
        DEPTH_VIDEO_RESOLUTION=(480,640)
        min_x,min_y,w,h = cv2.boundingRect(image)
        max_x=min_x+w
        max_y=min_y+h
        min_x=max(0,min_x-tollerance)
        min_y=max(0,min_y-tollerance)
        max_x=min(DEPTH_VIDEO_RESOLUTION[1],max_x+tollerance)
        max_y=min(DEPTH_VIDEO_RESOLUTION[0],max_y+tollerance)
        start=(min_x,min_y)
        end=(max_x,max_y)
        result = image[start[1]:end[1], start[0]:end[0]]
        percentage=max([(i,result.shape[i]/DEPTH_VIDEO_RESOLUTION[i]) for i in [0,1]],key=lambda x:x[1])
        if percentage[0]==0:
            result=cv2.resize(result,(int(result.shape[1]/percentage[1]),DEPTH_VIDEO_RESOLUTION[0]),cv2.INTER_AREA)
        else:
            result=cv2.resize(result,(DEPTH_VIDEO_RESOLUTION[1],int(result.shape[0]/percentage[1])),cv2.INTER_AREA)    
        return result           

    def calculate_humoments(self,im):
        # Threshold image
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _,im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
        
        im=self.get_roi(im)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        gradient = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, element)
        # Calculate Moments
        moment = cv2.moments(gradient)
        # Calculate Hu Moments
        hu_moments = cv2.HuMoments(moment)
        return hu_moments

    def depth_to_meter(self, depth):
        depth=depth.astype(float)
        try:
            return 1/((depth * 4 * -0.0030711016) + 3.3309495161)
        except:
            return 0.0    

    # just return mean of distances from points in cloud1 to their nearest neighbors in cloud2 
    def cloudAlignmentScoreDense(self, cloud1, cloud2):
        tree = KDTree(cloud2)
        N=cloud1.shape[0]
        accum=0.0
        result = tree.query(cloud1, k=1)
        for i,(dist, ind) in enumerate(zip(*result)):
            accum += dist[0]
        return accum/N

    def cloudAlignmentScoreDenseWithNormalsNormalized(self, cloud1, normals1, cloud2, normals2, relweight, dnormalize):
        tree = KDTree(cloud2)
        N=cloud1.shape[0]
        accum=0.0
        result = tree.query(cloud1, k=1)
        for i,(dist, ind) in enumerate(zip(*result)):
            dot = np.dot(normals1[i],normals2[ind[0]])
            if np.isnan(dot):
                continue
            accum += dist[0] / dnormalize
            accum += relweight*(1.0 - dot)      
        return accum/N

    def calculate_compactness_3d(self, points, pixel_length):   
        return points.shape[0] / pixel_length**2
        ''' 
        max_length = np.max(points,axis=0)[0]
        min_length = np.min(points,axis=0)[0]
        return points.shape[0] / (max(max_length-min_length, 0.0000001)**2)
        '''

    def calculate_symmetry_3d(self, points_np, normals, relweight=SYMMETRY_MEASURE_CLOUD_NORMALS_TRADEOFF):
        mins=points_np.min(axis=0)
        maxes=points_np.max(axis=0)
        ranges = maxes - mins
        ranges /= ranges.sum()
        score=0.0
        for i,vector in enumerate(np.array([[-1,1,1],[1,-1,1],[1,1,-1]])):
            dest=points_np*vector
            normdest=normals*vector
            overlap = self.cloudAlignmentScoreDenseWithNormalsNormalized(points_np, normals, dest, normdest, relweight, ranges[i])\
                    +self.cloudAlignmentScoreDenseWithNormalsNormalized(dest, normdest, points_np, normals, relweight, ranges[i])    
            score += ranges[i]*overlap
        return -score

    def calculate_global_convexity_3d(self, points):
        hull=ConvexHull(points)
        overlap= self.cloudAlignmentScoreDense(points, hull.points[hull.vertices])
        return -overlap

    def calculate_local_convexity_and_smoothness_3d(self, points, normals, NNradius=NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
        tree = KDTree(points)
        N=points.shape[0]
        score=0.0
        Hs=0.0
        bins=np.ones(NUMBINS)
        neighbors = tree.query_radius(points, NNradius)
        for i,(p1,n1,neighbors_current) in enumerate(zip(points,normals,neighbors)):
            if np.isnan(n1).any():
                N-=1
                continue
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

    def calculate_local_convexity_3d(self, points, normals, NNradius=NNRADIUS):
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

    def calculate_smoothness_3d(self, points, normals, NNradius=NNRADIUS, NUMBINS=SMOOTHNESS_MEASURE_NUMBINS):
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

    def pad_image(self, depth,result_shape=(480,640)):
        top=int((result_shape[0]-depth.shape[0])/2)
        bottom=result_shape[0]-top-depth.shape[0]
        left=int((result_shape[1]-depth.shape[1])/2)
        right=result_shape[1]-left-depth.shape[1]
        return np.pad(depth, ((top, bottom), (left, right)), 'constant')

    def cvtDepthColor2Cloud(self, depth):
        cameraMatrix = np.array(
            [[525., 0., 320.0],
            [0., 525., 240.0],
            [0., 0., 1.]])
        inv_fx = 1.0 / cameraMatrix[0, 0]
        inv_fy = 1.0 / cameraMatrix[1, 1]
        ox = cameraMatrix[0, 2]
        oy = cameraMatrix[1, 2]

        rows, cols = depth.shape
        cloud = np.zeros((depth.size, 3), dtype=np.float32)
        for y in range(rows):
            for x in range(cols):
                x1 = float(x)
                y1 = float(y)
                dist = depth[y][x]
                cloud[y * cols + x][0] = np.float32((x1 - ox) * dist * inv_fx)
                cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
                cloud[y * cols + x][2] = np.float32(dist)
        
        return pcl.PointCloud().from_array(cloud)  

    def depth_to_cloud(self, depth_original):
        depth=self.depth_to_meter(depth_original)
        depth=self.pad_image(depth)
        depth_original=self.pad_image(depth_original)
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
        xyz = np.float32(xyz)
        cloud = pcl.PointCloud()
        cloud.from_array(xyz)
        return cloud,xyz

    def get_cloud_and_normals(self, depth):
        point_cloud,cloud = self.depth_to_cloud(depth)
        # get points and normals
        ne = point_cloud.make_NormalEstimation()
        tree = point_cloud.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_RadiusSearch(0.05)
        normals = ne.compute()
        normals_point = normals.to_array()[:,0:3]
    
        return cloud,normals_point

    def get_mask(self, depth):
        mask=depth.copy()
        mask[mask>0]=255
        return mask

    def calculate_descriptors(self, depth):
        mask=self.get_mask(depth)
        # for avoiding flatten depth, otherwise convexhull raise error
        if np.min(depth[depth>0]) == np.max(depth):
            x=random.randrange(depth.shape[0])
            y=random.randrange(depth.shape[1])
            depth[x,y]+=1
        descriptors_2d=self.calculate_descriptors_2d(mask)
        descriptors_3d=self.calculate_descriptors_3d(depth)
        return descriptors_2d+descriptors_3d

    def calculate_descriptors_3d(self, depth):
        cloud,normals=self.get_cloud_and_normals(depth)

        compactness_3d=self.calculate_compactness_3d(cloud,depth.shape[0])
        symmetry_3d=self.calculate_symmetry_3d(cloud, normals)
        global_convexity_3d=self.calculate_global_convexity_3d(cloud)
        local_convexity_3d,smoothness_3d=self.calculate_local_convexity_and_smoothness_3d(cloud, normals)

        return [compactness_3d,symmetry_3d,global_convexity_3d,local_convexity_3d,smoothness_3d]

    def get_roi(self, image):
        min_x,min_y,w,h = cv2.boundingRect(image)
        max_x=min_x+w
        max_y=min_y+h
        return image[min_y:max_y,min_x:max_x]

    def calculate_descriptors_2d(self, mask):
        mask_roi=self.get_roi(mask)
        compactess_2d=self.calculate_compactess_2d(mask_roi)
        symmetry_2d=self.calculate_symmetry_2d(mask_roi)
        global_convexity_2d=self.calculate_global_convexity_2d(mask)
        histogram, uniqueness_2d=self.calculate_uniqueness_2d(mask)
        smoothness_2d=self.calculate_smoothness_2d(mask,histogram)
        return [compactess_2d,symmetry_2d,global_convexity_2d,uniqueness_2d,smoothness_2d]

    def calculate_compactess_2d(self, mask):
        pixels_on = cv2.countNonZero(mask)
        pixels = mask.shape[0] * mask.shape[1]
        return pixels_on/pixels    

    def calculate_symmetry_2d(self, mask):
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


    def calculate_global_convexity_2d(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=max(contours,key=lambda x: x.shape[0])
        hull = cv2.convexHull(contours)
        m=mask.copy()
        cv2.drawContours(m, [hull], -1, 150, 1)
        contours_pos=np.argwhere(m==150)
        points=np.argwhere(m==255)
        result=np.average(np.min(np.linalg.norm(contours_pos - points[:,None], axis=-1),axis=1))
        #normalize
        result/=max(mask.shape)/2
        return result

    def get_angle(self, v1,v2):
        angles=np.array([[135,120,90,60,45],
                        [150,135,90,45,30],
                        [180,180,0,0,0],
                        [210,225,270,315,330],
                        [225,240,270,300,315]])
        return (angles[v1[0],v1[1]]-angles[v2[0],v2[1]])%180                    

    def entropy(self, hist):
        return -sum([i*math.log(i) for i in hist])

    def calculate_uniqueness_2d(self, mask,show_hist=False):
        hist={}
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        m=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(m, contours, -1, (0,255,255), 1)
        t=3
        contours=max(contours,key=lambda x: x.shape[0])
        l=len(contours)
        vectors=[contours[(i+2)%l]-contours[(i-2)%l] for i in range(0,l,t)]
        l=len(vectors)
        
        
        for i in range(l):
            angle=self.get_angle(vectors[i][0],vectors[(i+1)%l][0])
            if angle in hist.keys():
                hist[angle]+=1
            else:
                hist[angle]=1
        if show_hist:
            print("I'm sorry, I'm not ready to plot histogram yet...")
            '''
            # At moment pyplot cause problem used with show_assistent=True, will be fixed in future
            import matplotlib.pyplot as plt
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
            '''          
        h=[i/l for i in hist.values()]  
        h2=[(k,v) for k,v in hist.items()]       
        return h2,self.entropy(h)

    def calculate_smoothness_2d(self, mask, histogram):
        X = np.array(histogram)
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)        
        return np.max(gm.means_[:,0])/180  #normalized  

def main():
    def get_image(filename,color=True, image_path=data_dir_images ):
        path=os.path.join(image_path,filename)
        if not color:
            im = cv2.imread(path,0)
        else: 
            im = cv2.imread(path)   
        return im 

    def apply_mask(mask,image):
        i=image.copy()
        if len(image.shape)==2:
            i[mask == 0]=0
        else:
            i[mask == 0]=np.array([0,0,0])    
        return i     

    from Grounding import round_list
    try:
        files = os.listdir(data_dir_images)
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)
    
    name="_".join(random.choice(files).split("_")[0:-1])
    depth=get_image(name+"_depthcrop.png",0)
    mask=get_image(name+"_maskcrop.png",0)
    depth=apply_mask(mask,depth)

    e=Shape_extractor()
    descriptors=e.extract(depth)   
    print("Shape descriptors: {}".format(round_list(descriptors)))    


if __name__=="__main__":
    main()            