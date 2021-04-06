if __package__:
    from Grounding.Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb
    from Grounding.Shape_extractor import Shape_extractor
    from Grounding.Texture_extractor import Texture_extractor
else:
    from Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb
    from Shape_extractor import Shape_extractor
    from Texture_extractor import Texture_extractor    
import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import pickle
import ast

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
    if type(l[0][0]) is list:
        return [([round(e,approx) for e in a],round(b,approx)) for (a,b) in l]
    else:  
        return [(a,round(b,approx)) for (a,b) in l]  

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
        if self.verbose:
            print("Intent: {}".format(intent))
            print("{}: {}\n{} features: {}\n".format(space,round_list(features[space][0]),space,round_list(features[space][1])))
            if space=="color":
                self.print_colors(features['color'][1])        
        return features[space][0]

    def extract(self,scan,space_name):
        # TODO extract only feature needed
        image,depth,merged=scan
        features={}
        features['color']=self.color_extractor.extract(merged)
        features['shape']=self.shape_extractor.extract(image)
        features['texture']=self.texture_extractor.extract(image)
        features['general']=sorted([(i, random.random()) for i in ["palla da tennis", "uovo", "banana", "pera", "anguria", "anguilla"]],key=lambda x:x[1])
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
                    space.space[knowledge_name]=ConvexHull(points,incremental=True)
 

    def learn(self, scan, intent, label):
        # TODO general learn
        image,depth,merged=scan
        space_label=intent[:-9]
        features=self.extract(scan,space_label)
        feature=features[space_label][1][0][0]
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

    def show_space(self,fig=None):
        if not fig:
            fig = plt.figure(num="Tensor Spaces",figsize=(15,5))
        ax = fig.add_subplot(1,3,self.space_index+1, projection="3d")
        ax.set_title(self.space_label)

        for label,hull in self.space.items():
            for s in hull.simplices:
                color=[i/255 for i in cielab_to_rgb(hull.points[0])]
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(hull.points.T[0], hull.points.T[1], hull.points.T[2], "o", color=color)
                ax.plot(hull.points[s, 0], hull.points[s, 1], hull.points[s, 2],"-", color=color)
                ax.text(hull.points[0][0],hull.points[0][1],hull.points[0][2], label, size=10, zorder=1, color='k')

        # Make axis label
        if self.space_label=="color":
            axis_label={"x":"L", "y":"A", "z":"B"}
        else:
            axis_label={"x":"x", "y":"y", "z":"z"}    
        for i,label in axis_label.items():
            eval("ax.set_{:s}label('{:s}')".format(i, label))
       

def main(mod):
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
    depth = None
    merged=None
    g=Grounding(True)
    if mod=="classify":
        g.classify((img,depth,img),"color_query")
    elif mod=="learning":    
        g.learn((img,depth,img),"color_training",image[:-4].split("-")[0])
        g.spaces.show_spaces()

if __name__=="__main__":
    main("learning")           


