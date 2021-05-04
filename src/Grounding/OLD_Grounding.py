import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import pickle
import itertools
import math
import ast
if __package__:
    from Grounding.Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb
    from Grounding.Shape_extractor import Shape_extractor
    from Grounding.Texture_extractor import Texture_extractor
else:
    from Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb
    from Shape_extractor import Shape_extractor
    from Texture_extractor import Texture_extractor 

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_grounding = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_grounding = os.path.dirname(__file__)
    data_dir_knowledge = os.path.join(data_dir_grounding,"knowledge")
    data_dir_images =os.path.join(data_dir_grounding,"..","..","Media","Images")

def round_list(l):
    if not l:
        return None
    approx=4
    if not hasattr(l[0], '__getitem__'):
        return [round(a,approx) for a in l]
    elif not type(l[0][0]) is list:
        return [(a,round(b,approx)) for (a,b) in l]      
    else:
        return [([round(e,approx) for e in a],round(b,approx)) for (a,b) in l]
      

class Grounding:
    def __init__(self,verbose=False):
        self.verbose=verbose
        self.color_extractor=Color_extractor()
        self.shape_extractor=Shape_extractor()
        self.texture_extractor=Texture_extractor()
        self.load_knowledge()
        
    def classify(self, scan, intent):
        space=intent[:-6]
        features=self.extract(scan,space)
        labels=self.classify_features(features,space)
        if self.verbose:
            if space=="shape":
                features[space]=features[space].tolist()
            print("Intent: {}".format(intent))
            print("\n{}: {}\n{} features: {}\n".format(space,round_list(labels),space,round_list(features[space])))
            if space=="color":
                self.print_colors(features['color'])        
        return labels

    def classify_features(self,features,space_name):
        if space_name=="general":
            return sorted([(i, random.random()) for i in ["palla da tennis", "uovo", "banana", "pera", "anguria", "anguilla"]],key=lambda x:x[1])
        feature=features[space_name]
        if space_name=="color":
            feature=feature[0][0] # Using only the feature more relevant
        if space_name=="shape":
            feature=feature[:3] # Using only the feature more relevant    
        distances=self.spaces.spaces[space_name].classify(feature) 
        if self.verbose:
            self.spaces.spaces[space_name].show_space(points=[feature])
            plt.show()
        classifications=[]
        probability=0
        for l,d in distances:
            p=1/(d+0.001)
            probability+=p
            classifications.append((l,p))
        classifications=[(l,d/probability) for l,d in classifications]
        classifications.sort(key=lambda x:x[1],reverse=True)     
        return classifications

    def extract(self,scan,space_name):
        image,depth,merged=scan
        features={}
        if space_name in ["color","general"]:
            features['color']=self.color_extractor.extract(merged)
        if space_name in ["shape","general"]:   
            features['shape']=self.shape_extractor.extract(merged)
        if space_name in ["texture","general"]:     
            features['texture']=self.texture_extractor.extract(image)
        return features

    def load_knowledge(self):
        space_names = [ f.name for f in os.scandir(data_dir_knowledge) if f.is_dir() ] # ["color","shape","texture"]
        self.spaces = Tensor_spaces(space_names)
        for space_label,space in self.spaces.spaces.items():
            folder = os.path.join(data_dir_knowledge,space_label)
            knowledge_files = os.listdir( folder )
            knowledge_files=[i for i in knowledge_files if i.endswith(".txt")]
            for knowledge_file in knowledge_files:    
                path = os.path.join(folder,knowledge_file)
                knowledge_name = knowledge_file[:-4]
                points=[]
                with open(path, "r") as f:
                    for line in f.readlines():
                        line=line.strip()
                        line=line.split("#")[0]
                        points.append(ast.literal_eval(line))
                    if space_label=="shape":
                        points=[p[:3] for p in points]    
                    space.space[knowledge_name]=ConvexHull(points,incremental=True)
 

    def learn(self, scan, intent, label):
        # TODO general learn
        image,depth,merged=scan
        space_label=intent[:-9]
        features=self.extract(scan,space_label)
        feature=features[space_label][0][0]
        self.spaces.insert(space_label,label,feature)

        

    def print_colors(self,color_list):
        color_matrix=None
        for c in color_list:
            color=cielab_to_cv2lab(c[0])
            if color_matrix is not None:
                color_matrix=np.concatenate((color_matrix,np.full((20,20,3),np.array(color))),axis=0)
            else:
                color_matrix=np.full((20,20,3),np.array(color))   
        plt.figure(num="Colors")
        plt.imshow(cv2.cvtColor(color_matrix.astype(np.uint8), cv2.COLOR_LAB2RGB))
        plt.show()    


class Tensor_spaces:
    def __init__(self, names):
        self.spaces={l:Tensor_space(i,l) for i,l in enumerate(names)}  

    def insert(self,space_label,label,point):
       self.spaces[space_label].insert(label,point) 

    def show_spaces(self):
        fig = plt.figure(num="Tensor Spaces", figsize=(15,5))
        for space in self.spaces.values():
            space.show_space(fig)
        plt.show()    

class Tensor_space:
    def __init__(self,space_index,space_label):
        self.space={}
        self.space_label=space_label
        self.space_index=space_index

    def classify(self,feature):
        distances=[]
        for label,hull in self.space.items():
            d=self.distance_hull_point(feature,hull)
            if d==0:
                return [(label,d)]
            distances.append((label,d)) 
        return sorted(distances,key=lambda x:x[1])    

    def point_in_hull(self, point, hull, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)

    def get_corners(self,vertices):
        return list(itertools.combinations(vertices,2))

    def distance_hull_point(self, point, hull):
        if self.point_in_hull(point, hull):
            return 0.0
        return min([self.distance_point_face(point,(hull.points[s[0]],s[1])) for s in zip(hull.simplices,hull.equations)])    

    def project(self, point, plane, tolerance=1e-12):
        if abs(np.dot(plane[:-1], point) + plane[-1]) <= tolerance:
            return point
        t = -(plane[-1] + np.dot(plane[:-1], point))/(np.sum(plane[:-1]**2))
        return point + plane[:-1]*t  

    def triangle_area(self,p1,p2,p3):
        segments=list(itertools.combinations([p1,p2,p3],2))
        segments_len=[np.linalg.norm(s[0]-s[1]) for s in segments]
        semiperimeter=sum(segments_len)/2
        result=semiperimeter
        for l in segments_len:
            result*=semiperimeter-l
        return math.sqrt(result)
    
    def in_triangle(self,point_projection,vertices,corners=None,tolerance=1e-12):
        if not corners:
            corners=self.get_corners(vertices)
        total_area=self.triangle_area(*vertices)
        sub_triangle_areas=[(self.triangle_area(point_projection,p1,p2)/total_area) for p1,p2 in corners]
        return all(0<=t<=1 for t in sub_triangle_areas) and 1-tolerance<=sum(sub_triangle_areas)<=1+tolerance

    def distance_point_corner(self,p,corner):
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

    def distance_point_plane(self,point,plane):
        return abs(plane[-1] + np.dot(plane[:-1], point))/math.sqrt(np.sum(plane[:-1]**2))

    def distance_point_face(self,point,face):
        vertices,plane=face
        point_projection=self.project(point,plane)
        corners=self.get_corners(vertices)
        if self.in_triangle(point_projection,vertices,corners):
            return self.distance_point_plane(point,plane)
        else:
            return min([self.distance_point_corner(point,c) for c in corners])             

    def insert(self,label,point):
        if label in self.space.keys(): # label just learned
            #old_points=np.array([self.space[label].points[j] for j in self.space[label].vertices])
            self.space[label].add_points([point])
        else: # new label
            #old_points=np.array()
            self.space[label]=ConvexHull(self.augument(point),incremental=True)
        new_points=[self.space[label].points[j] for j in self.space[label].vertices]
        self.save_knowledge(label,new_points)
        '''
        comparison=True
        if len(old_points)!=len(new_points):
            comparison=False
        else:
            for i in range(len(old_points)):
                if not (old_points[i]==new_points[i]).all():
                    comparison=False
        if not comparison:
            self.save_knowledge(label,new_points)
        '''
               

    def save_knowledge(self,label,points):
        with open(os.path.join(data_dir_knowledge,self.space_label,label+".txt"), "w") as f:
            for point in points:
                f.write(str(point.tolist())+"\n")      
               
    def augument(self,point):
        return [self.shift_point(point,axis,direction=direction) for axis in range(len(point)) for direction in [1,-1]]

    def shift_point(self,point,axis,increment=0.00001,direction=1):
        p=point.copy()
        p[axis]+=increment*direction
        return p

    def show_space(self,fig=None,points=[]):
        if not fig:
            fig = plt.figure(num="Tensor Space",figsize=(8,8))
            ax = fig.add_subplot(1,1,1, projection="3d")
        else:    
            ax = fig.add_subplot(1,3,self.space_index+1, projection="3d")
        ax.set_title(self.space_label)

        for label,hull in self.space.items():
            color=self.get_color_plot(hull.points[0])
            for s in hull.simplices:  
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(hull.points.T[2], hull.points.T[1], hull.points.T[0], "o", color=color)
                ax.plot(hull.points[s, 2], hull.points[s, 1], hull.points[s, 0],"-", color=color)
                ax.text(hull.points[0][2],hull.points[0][1],hull.points[0][0], label, size=10, zorder=1, color='k')
        for i,p in enumerate(points):
            color=self.get_color_plot(p)
            ax.plot(p[2], p[1], p[0], "o", markersize=20, color=color)
            ax.text(p[2], p[1], p[0], "point_"+str(i), size=20, zorder=1, color='k')

        # Make axis label
        if self.space_label=="color":
            axis_label={"x":"B", "y":"A", "z":"L"}
        else:
            axis_label={"x":"x", "y":"y", "z":"z"}    
        for i,label in axis_label.items():
            eval("ax.set_{:s}label('{:s}')".format(i, label))
        ax.invert_xaxis()    

    def get_color_plot(self,p):
        if self.space_label=="color":
            return [i/255 for i in cielab_to_rgb(p)]
        return [random.random(),random.random(),random.random()]        
       

def main(mod,space):
    try:
        images = os.listdir( data_dir_images )
        images=[i for i in images if i.endswith(".jpg")]
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)
    image=random.choice(images) 
    print("Image: {}".format(image))   
    path = os.path.join(data_dir_images,image)
    img = cv2.imread(path)
    if space=="shape":
        img=~img
    depth = None
    merged=None
    g=Grounding(True)
    if mod=="classify":
        g.classify((img,depth,img),space+"_query")
    elif mod=="learning":    
        g.learn((img,depth,img),space+"_training",image[:-4].split("-")[0])
        g.spaces.show_spaces()

if __name__=="__main__":
    main("classify","shape")           


