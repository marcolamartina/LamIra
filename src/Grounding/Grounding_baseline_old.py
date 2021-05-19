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
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))

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

    def classify_general_features(self,features):
        MAX_ALTERNATIVE=2
        alternative_list=[f[:MAX_ALTERNATIVE] for f in features]
        combs=list(itertools.product(*alternative_list))
        components=[]
        for c in combs:
            prob=0
            labels=[]
            for e in c:
                prob+=e[1]
                labels.append(e[0])
            components.append((labels,prob))
        components.sort(key=lambda x:x[1],reverse=True)        
        return [tuple(c[0]) for c in components]

    def extract_general_features(self,features):
        features_general=[self.classify_features(features,s) for s in features.keys() if s!="general"]
        feature=self.classify_general_features(features_general)
        return feature       

    def classify_features(self,features,space_name): 
        feature=features[space_name]   
        distances=self.spaces.spaces[space_name].classify(feature)       
        classifications=[]
        x1,y1=0,1
        x2,y2=max([i[1] for i in distances]),0
        x2*=1.1
        func=lambda x:y1+(x-x1)*(y2-y1)/(x2-x1)
        classifications=[(l,func(d)) for l,d in distances]
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
        if space_name=="general":
            features['general']=self.extract_general_features(features)
        if self.verbose:
            if space_name=="general":
                print("{} features: {}".format(space_name,features[space_name]))
            else:
                print("{} features: {}".format(space_name,round_list(features[space_name])))    
        return features
 

    def learn(self, scan, intent, label):
        color_masked,depth_masked=scan
        space_label=intent[:-9]
        features=self.extract(scan,space_label)
        feature=features[space_label]
        self.spaces.insert(space_label,label,feature)
        return features
  

    def load_knowledge(self):
        space_names = [ f.name for f in os.scandir(data_dir_knowledge) if f.is_dir() ] # ["color","shape","texture","general"]
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
            if space_label=="general":
                for label, features in space.space.items():
                    for feature in features:  
                        if feature in space.space_inv.keys(): # features tuple just added
                            space.space_inv[feature].append(label)
                        else: # new features tuple
                            space.space_inv[feature]=[label]       
        

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
        self.spaces={l:Tensor_space(l) for l in names if l!="general"}
        self.spaces["general"]=Conseptual_space("general") 

    def insert(self,space_label,label,point):
       self.spaces[space_label].insert(label,point) 
   

class Tensor_space:
    def __init__(self,space_label):
        self.space={}
        self.space_label=space_label

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


class Conseptual_space(Tensor_space):
    def __init__(self,space_label):
        super().__init__(space_label)
        self.space_inv={}

    def classify(self,features):
        result={}
        for w,feature in enumerate(features):
            if feature in self.space_inv.keys():
                for label in self.space_inv[feature]:
                    if label in result.keys():
                        result[label]-=1/(w+0.000001)
                    else:    
                        result[label]=w
                min_w=min(result.values())
                result={label:min_w+weight for label,weight in result.items()}    
        if len(result):
            return sorted(list(result.items()),key=lambda x:x[1])
        # no matching
        # TODO to fix

        best_label="Niente"
        best_color=""
        best_shape=""
        best_texture=""
        color,shape,texture=features[0]
        best_weight=0
        for f,l in self.space_inv.items():
            label=l[0]    
            weight=0
            color_p,shape_p,texture_p=f
            if color_p==color:
                weight+=1
            if shape_p==shape:
                weight+=2 # shape is more important
            if texture_p==texture:
                weight+=1    
            if weight>best_weight:
                best_label=label
                best_weight=weight
                best_color,best_shape,best_texture=f
            if weight>=3:
                break    
        result=[best_label]
        if best_color!=color:
            result.append((color,best_color))
        if best_shape!=shape:
            result.append((shape,best_shape))
        if best_texture!=texture:
            result.append((texture,best_texture))  
        # return "unsure",result
        print("Not matched")
        return [(best_label,1)]

    def insert(self,label,features):
        if label in self.space.keys(): # label just learned
            self.space[label].append(features)
        else: # new features tuple
            self.space[label]=features
        for feature in features:        
            if feature in self.space_inv.keys(): # features tuple just learned
                self.space_inv[feature].append(label)
            else: # new features tuple
                self.space_inv[feature]=[label]
        self.save_knowledge(label)    


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

def learn_features():
    import time

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

    path = os.path.dirname(__file__)    
    path = os.path.join(path,"..","..","Datasets")
    path_ds = os.path.join(path,"rgbd-dataset")
    path_descriptors = os.path.join(path,"dataset-descriptors")
    g=Grounding(False)
    stride=10
    start_index=0
    checkpoint=161
    end=210
    with open(path_ds+"/training.txt", "r") as f:
        lines_list=f.readlines()
        number_of_object=len(lines_list)
        for number_line,line in enumerate(lines_list):
            if number_line<=checkpoint:
                continue
            if number_line>=end:
                break    
            line=line.strip()
            line=line.split(":")
            name = line[0].rsplit("_", 1)[0]
            features = line[1].split(";")
            shape,color,texture=features
            path_images = os.path.join(path_ds, name, line[0])

            try:
                images = os.listdir( path_images )
                images = [i.rsplit("_", 1)[0] for index, i in enumerate(images) if i.endswith("_crop.png") and index%stride==start_index]
                images = sorted(images)
            except FileNotFoundError:
                print("{}: No such file or directory".format(path_images))
                os._exit(1)  
            
            #print(images)
            #os._exit(1) #da rimuovere per fare tutte
            l=len(images)-1
            starting_time = time.time()
            print("Starting learn image directory: {} at {}".format(line[0], time.asctime().split(" ")[3]))  
            for index,i in enumerate(images):
                #if i.endswith("_crop.png"):
                #i=i.rsplit("_", 1)[0]
                depth=get_image(i+"_depthcrop.png",0, image_path=path_images)
                img=get_image(i+"_crop.png", image_path=path_images)
                mask=get_image(i+"_maskcrop.png",0, image_path=path_images)
                depth=apply_mask(mask,depth)
                img=apply_mask(mask,img)
                print("{:.2f}%".format(index*100/l))   
                
                #g.spaces.insert("general",label,feature)
                try:
                    color_descriptor = g.learn((img, depth),"color_training",color)
                    color_descriptor = color_descriptor["color"]
                    shape_descriptor = g.learn((img, depth),"shape_training",shape)
                    shape_descriptor = shape_descriptor["shape"]
                    texture_descriptor = g.learn((img,depth),"texture_training",texture)
                    texture_descriptor = texture_descriptor["texture"]
                except:
                    print("Error")
                    continue
                descr_path=os.path.join(path_descriptors, name, line[0])
                with open(descr_path+"/"+i+".txt", "w+") as f:
                    f.writelines(str(color_descriptor)+"\n")
                    f.writelines(str(shape_descriptor)+"\n")
                    f.writelines(str(texture_descriptor)+"\n")
                
                #os._exit(1) #da rimuovere per fare tutte

            print("100% \nFinished learn image directory {}/{}: {} at {} in {:.2f}m ".format(number_line,number_of_object,line[0], time.asctime().split(" ")[3], (time.time()-starting_time)/60))   

def learn_knowledge():
    import ast
    from glob import glob

    def translate(name):
        translate_dict={"apple":"mela",
                        "ball":"palla",
                        "bell pepper":"peperone",
                        "binder":"raccoglitore",
                        "bowl":"ciotola",
                        "calculator":"calcolatrice",
                        "camera":"fotocamera",
                        "cell phone":"telefono",
                        "cereal box":"scatola",
                        "coffee mug":"tazza",
                        "comb":"spazzola",
                        "dry battery":"batteria",
                        "flashlight":"torcia",
                        "food box":"scatola",
                        "food can":"lattina",
                        "food cup":"barattolo",
                        "food jar":"barattolo",
                        "garlic":"aglio",
                        "lemon":"limone",
                        "lime":"lime",
                        "onion":"cipolla",
                        "orange":"arancia",
                        "peach":"pesca",
                        "pear":"pera",
                        "potato":"patata",
                        "tomato":"pomodoro",
                        "soda can":"lattina",
                        "marker":"pennarello",
                        "plate":"piatto",
                        "notebook":"quaderno",
                        "keyboard":"tastiera",
                        "glue stick":"colla",
                        "sponge":"spugna",
                        "toothpaste":"dentifricio",
                        "toothbrush":"spazzolino"
                        }
        try:
            return translate_dict[name]
        except:
            return name    
    
    path = os.path.dirname(__file__)    
    path = os.path.join(path,"..","..","Datasets")
    path_ds = os.path.join(path,"rgbd-dataset")
    path_descriptors = os.path.join(path,"dataset-descriptors")
    g=Grounding(False)

    for filename in glob(path_descriptors+'/**', recursive=True):
        if os.path.isfile(filename) and filename.endswith(".txt"):
            name=" ".join(filename.split("_")[:-3])
            name=translate(name)
            with open(filename, "r") as f:
                features=[ast.literal_eval(line) for line in f.readlines()]
                features_dict={"color":features[0],"shape":features[1],"texture":features[2]}
                try:
                    features_label=g.extract_general_features(features_dict)
                except:
                    continue    
                g.spaces.insert("general",name,features_label)

                

if __name__=="__main__":
    main("classify","general")
    #learn_features()           
