import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import pickle
import itertools
import math
import ast
import pickle
if __package__:
    test=False
    from Grounding.Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb
    from Grounding.Shape_extractor import Shape_extractor
    from Grounding.Texture_extractor import Texture_extractor
else:
    test=True
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
    data_dir_images = os.path.join(data_dir_grounding,"..","..","Datasets","rgbd-dataset")
    data_dir_images = os.path.join(data_dir_images,random.choice(os.listdir(data_dir_images)))
    data_dir_images = os.path.join(data_dir_images,random.choice(os.listdir(data_dir_images)))

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
        if self.verbose:
            print("Intent: {}".format(intent))
        features=self.extract(scan,space)
        labels=self.classify_features(features,space)
        if self.verbose and space=="color" and test:
            self.print_colors(features['color'])        
        return labels

    def classify_features(self,features,space_name):
        if space_name=="general":
            return sorted([(i, random.random()) for i in ["palla da tennis", "uovo", "banana", "pera", "anguria", "anguilla"]],key=lambda x:x[1])
        feature=features[space_name]   
        distances=self.spaces.spaces[space_name].classify(feature) 
        classifications=[]
        probability=0
        for l,d in distances:
            p=1/(d+0.001)
            probability+=p
            classifications.append((l,p))
        classifications=[(l,d/probability) for l,d in classifications]
        classifications.sort(key=lambda x:x[1],reverse=True) 
        if self.verbose:
            print("{}: {}".format(space_name,round_list(classifications)))
        return classifications

    def extract(self,scan,space_name):
        color_masked,depth_masked=scan
        features={}
        #features[space_name]=getattr(self, space_name+"_extractor").extract(color_masked)
        if space_name in ["color","general"]:
            features['color']=self.color_extractor.extract(color_masked)
        if space_name in ["shape","general"]:   
            features['shape']=self.shape_extractor.extract(depth_masked)
        if space_name in ["texture","general"]:     
            features['texture']=self.texture_extractor.extract(color_masked)
        print("{} features: {}".format(space_name,round_list(features[space_name])))    
        return features
 

    def learn(self, scan, intent, label):
        # TODO general learn
        color_masked,depth_masked=scan
        space_label=intent[:-9]
        features=self.extract(scan,space_label)
        feature=features[space_label]
        self.spaces.insert(space_label,label,feature)
  

    def load_knowledge(self):
        space_names = [ f.name for f in os.scandir(data_dir_knowledge) if f.is_dir() ] # ["color","shape","texture"]
        self.spaces = Tensor_spaces(space_names)
        for space_label,space in self.spaces.spaces.items():
            folder = os.path.join(data_dir_knowledge,space_label)
            knowledge_files = os.listdir( folder )
            knowledge_files=[i for i in knowledge_files if i.endswith(".pickle")]
            for knowledge_file in knowledge_files:    
                path = os.path.join(folder,knowledge_file)
                knowledge_name = knowledge_file[:-7]
                with open(path, "rb") as f:    
                    space.space[knowledge_name]=pickle.loads(f.read()) 


    def print_colors(self,color_list):
        color_matrix=None
        for c in [color_list[i:i + 3] for i in range(0, len(color_list), 3)]:
            color=cielab_to_cv2lab(c)
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
   

class Tensor_space:
    def __init__(self,space_index,space_label):
        self.space={}
        self.space_label=space_label
        self.space_index=space_index

    def save_knowledge(self,label):
        with open(os.path.join(data_dir_knowledge,self.space_label,label+".pickle"), "wb") as f:
            s = pickle.dumps(self.space[label])
            f.write(s)

    def classify(self,feature):
        distances=[]
        for label,tree in self.space.items():
            d=self.distance(feature,tree)
            if d==0:
                return [(label,d)]
            distances.append((label,d)) 
        return sorted(distances,key=lambda x:x[1])    

    def insert(self,label,point):
        if label in self.space.keys(): # label just learned
            points=self.space[label].get_arrays()[0]
            np.append(points,np.array([point]))
            self.space[label]=KDTree(points)
        else: # new label
            self.space[label]=KDTree(np.array([point]))
        self.save_knowledge(label)    

    def in_ellipsoid(self, point, ellipsoid, centroid):
        if 0 in ellipsoid:
            return False
        return (np.square(point-centroid)/np.square(ellipsoid)).sum()<1        

    def distance(self,feature,tree):
        points=np.array(tree.get_arrays()[0])
        mins=points.min(axis=0)
        maxes=points.max(axis=0)
        centroid=points.mean(axis=0)
        ellipsoid = (maxes - mins)/2
        if self.in_ellipsoid(feature, ellipsoid, centroid):
            return 0
        dist, _ = tree.query(np.array([feature]), k=1)
        return dist[0][0]

def main(mod,space):
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

    try:
        files = os.listdir(data_dir_images)
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)
    name="_".join(random.choice(files).split("_")[0:-1])
    depth=get_image(name+"_depthcrop.png",0)
    img=get_image(name+"_crop.png")
    mask=get_image(name+"_maskcrop.png",0)
    depth=apply_mask(mask,depth)
    img=apply_mask(mask,img)

    print("Image: {}".format(name))   
    g=Grounding(True)
    if mod=="classify":
        g.classify((img,depth),space+"_query")
    elif mod=="learning":    
        g.learn((img,depth),space+"_training"," ".join(name.split("_")[:-3]))

if __name__=="__main__":
    main("classify","color")           


