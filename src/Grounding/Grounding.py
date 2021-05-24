import random
import os
from types import CodeType
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
import ast
from glob import glob
from sklearn.ensemble import RandomForestClassifier

if __package__:
    test=False
    from Grounding.Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb, normalize_color, print_colors
    from Grounding.Shape_extractor import Shape_extractor
    from Grounding.Texture_extractor import Texture_extractor
else:
    test=True
    from Color_extractor import Color_extractor, cielab_to_cv2lab, cv2lab_to_cielab, cielab_to_rgb, normalize_color, print_colors
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
    data_dir_base_knowledge = os.path.join(data_dir_grounding,"base_knowledge")
    data_dir_images = os.path.join(data_dir_grounding,"..","..","Datasets","rgbd-dataset")
    data_dir_images_captured = os.path.join(data_dir_images)
    data_dir_images_captured = os.path.join(data_dir_images_captured,random.choice([f.name for f in os.scandir(data_dir_images_captured) if f.is_dir() and not f.name.startswith("_")]))
    data_dir_images_captured = os.path.join(data_dir_images_captured,random.choice([f.name for f in os.scandir(data_dir_images_captured) if f.is_dir() and not f.name.startswith("_")]))
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


    def translate(self, name):
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

    def create_base_knowledge(self,overwrite=False):
        X={"general":[], "color":[], "shape":[],"texture":[]}
        y={"general":[], "color":[], "shape":[],"texture":[]}
        data_dir = data_dir_base_knowledge+"/Data"
        exclusion_list=["binder","camera","cell phone","dry battery"]
        file_list=glob(data_dir+'/**', recursive=True)
        number_of_files=len(file_list)
        with open(data_dir_grounding+"/dictionary.pickle","rb") as f:
            dictionary=pickle.load(f)
            for j,filename in enumerate(file_list):
                if os.path.isfile(filename) and filename.endswith(".txt"):
                    print("{:.2f}%".format(j*100/number_of_files))
                    name=" ".join(filename.split("_")[:-3]).rsplit("/", 1)[1]
                    if name in exclusion_list:
                        continue
                    name=self.translate(name)
                    folder=filename.split("/")[-2]
                    with open(filename, "r") as f:
                        features=[]
                        try:
                            lines=f.readlines()
                            for line in lines:
                                features.append(ast.literal_eval(line))
                            if len(features)==3:        
                                color,shape,texture=features
                                shape_name,color_name,texture_name=dictionary[folder]   
                                color=normalize_color(color)
                                general=color+shape+texture
                                general_name=name
                                for k,l in X.items():
                                    l.append(locals()[k])
                                for k,l in y.items():
                                    l.append(locals()[k+"_name"])    
                        except:
                            print("Error in {}".format(filename))
                            print(lines)
                            continue
            for k,l in X.items():
                X[k]=np.array(l)
                np.save(data_dir_base_knowledge+"/X_"+k+".npy",X[k])
                if overwrite:
                    np.save(data_dir_knowledge+"/"+k+"/X_"+k+".npy",X[k])
            for k,l in y.items():
                y[k]=np.array(l)   
                np.save(data_dir_base_knowledge+"/y_"+k+".npy",y[k])
                if overwrite:
                    np.save(data_dir_knowledge+"/"+k+"/y_"+k+".npy",y[k])       
                           
        
    def classify(self, scan, intent):
        space=intent[:-6]
        if self.verbose:
            print("Intent: {}".format(intent))
        features=self.extract(scan,space)
        labels=self.classify_features(features,space)
        if self.verbose and space=="color" and test:
            print_colors(features['color'])         
        return labels

    def extract_general_features(self,features):
        return normalize_color(features["color"])+features["shape"]+features["texture"]      

    def classify_features(self,features,space_name): 
        feature=features[space_name]   
        if space_name=="general":
            probabilities=self.spaces.spaces[space_name].classify(feature)
            if self.verbose:
                print("{}: {}".format(space_name,round_list(probabilities)))
            return probabilities
        else:    
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
            if True or space_label=="general":
                if "X_"+space_label+".npy" in knowledge_files:
                    
                    space.space["X"]=np.load(os.path.join(folder,"X_"+space_label+".npy"))
                if "y_"+space_label+".npy" in knowledge_files:
                    space.space["y"]=np.load(os.path.join(folder,"y_"+space_label+".npy"))         
                space.fit()
            else:    
                for knowledge_file in knowledge_files:
                    if knowledge_file.endswith(".pickle"):    
                        path = os.path.join(folder,knowledge_file)
                        knowledge_name = knowledge_file[:-7]
                        with open(path, "rb") as f:
                            space.space[knowledge_name]=pickle.loads(f.read())       


class Tensor_spaces:
    def __init__(self, names):
        #self.spaces={l:Tensor_space(l) for l in names if l!="general"}
        #self.spaces["general"]=Conseptual_space("general")
        self.spaces={l:Conseptual_space(l) for l in names} 

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

    def classify(self,feature,limit=4):
        distances=[]
        for label,tree in self.space.items():
            d=self.distance(feature,tree)
            if d==0:
                return [(label,d)]
            distances.append((label,d)) 
        return sorted(distances,key=lambda x:x[1])[:limit]    

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

    def save_knowledge(self,label):
        folder = os.path.join(data_dir_knowledge,self.space_label)
        np.save(os.path.join(folder,"X_"+self.space_label+".npy"),self.space["X"])
        np.save(os.path.join(folder,"y_"+self.space_label+".npy"),self.space["y"])


    def classify(self,features,limit=4):
        prob=self.clf.predict_proba(np.array([features]))[0]
        index=np.flip(np.argsort(prob))[:limit]
        labels=self.clf.classes_[index]
        prob=prob[index]
        coef=sum(prob)
        result=[(l,p/coef) for l,p in zip(labels,prob)]
        return result

    def insert(self,label,features):
        self.space["X"]=np.append(self.space["X"],np.array([features]), axis=0)
        self.space["y"]=np.append(self.space["y"],np.array([label]), axis=0)
        self.fit()
        self.save_knowledge(label)    

    def fit(self):
        self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=30)
        self.clf.fit(self.space["X"],self.space["y"])

def main(mod,space,captured=False):
    def get_image(filename,color=True, captured=False ):
        if captured:
            image_path=data_dir_images_captured
        else:
            image_path=data_dir_images
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
        if captured:
            files = os.listdir(data_dir_images_captured)
        else:
            files = os.listdir(data_dir_images)
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)
    g=Grounding(False)

    #If you want to change and update base knowledge files
    create_dictionary()
    g.create_base_knowledge(overwrite=True)

    filename=random.choice(files)    
    name="_".join(filename.split("_")[0:-1])
    depth=get_image(name+"_depthcrop.png",0,captured)
    img=get_image(name+"_crop.png",1,captured)
    mask=get_image(name+"_maskcrop.png",0,captured)
    depth=apply_mask(mask,depth)
    img=apply_mask(mask,img)
    if captured:
        print("Image: {}/{}".format(data_dir_images_captured.rsplit("/",1)[1],filename)) 
    else:    
        print("Image: {}".format(name))   
    if mod=="classify":
        for l in ["color","shape","texture","general"]:
            print(l,round_list(g.classify((img,depth),l+"_query")))
    elif mod=="learning":
        if captured:
            name_obj=" ".join(data_dir_images_captured.rsplit("/",1)[1].split("_")[:-1])
        else:
            name_obj=" ".join(name.split("_")[:-3])
        print(name_obj)
        print(round_list(g.learn((img,depth),space+"_training",name_obj)[space]))

def create_dictionary():
    with open(data_dir_grounding+"/training.txt", "r") as f:
        lines_list=f.readlines()
        dictionary = {}
        for line in lines_list: 
            line=line.strip()
            line=line.split(":")
            features = line[1].split(";")
            dictionary[line[0]]=features
              
    with open(data_dir_grounding+"/dictionary.pickle","wb") as f:
            o=pickle.dumps(dictionary)
            f.write(o)  

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
    checkpoint=-1
    end=2100
    with open(os.path.dirname(__file__)+"/training.txt", "r") as f:
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

            print("100%\nFinished learn image directory {}/{}: {} at {} in {:.2f}m ".format(number_line,number_of_object,line[0], time.asctime().split(" ")[3], (time.time()-starting_time)/60))              

def learn_knowledge():
    import ast
    from glob import glob   
    
    path = os.path.dirname(__file__)    
    path = os.path.join(path,"..","..","Datasets")
    path_descriptors = os.path.join(path,"dataset-descriptors")
    g=Grounding(False)

    for filename in glob(path_descriptors+'/**', recursive=True):
        if os.path.isfile(filename) and filename.endswith(".txt"):
            name=" ".join(filename.rsplit("/",1)[1].split("_")[:-3])
            name=g.translate(name)
            with open(filename, "r") as f:
                features=[ast.literal_eval(line) for line in f.readlines()]
                features_dict={"color":features[0],"shape":features[1],"texture":features[2]}
                try:
                    features_label=g.extract_general_features(features_dict)
                except:
                    continue    
                g.spaces.insert("general",name,features_label)

def learn_color():
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
   
    path = os.path.join(data_dir_grounding,"..","..","Datasets")
    path_ds = os.path.join(path,"rgbd-dataset")
    #path_ds = os.path.join(data_dir_grounding,"..","..","Datasets","rgbd-dataset")
    path_descriptors = os.path.join(data_dir_grounding,"base_knowledge","Data")
    g=Grounding(False)
    stride=10
    start_index=0
    checkpoint=-1
    end=10000
    with open(os.path.dirname(__file__)+"/training.txt", "r") as f:
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
            #features = line[1].split(";")
            #shape,color,texture=features
            path_images = os.path.join(path_ds, name, line[0])
            try:
                images = os.listdir( path_images )
                images = [i.rsplit("_", 1)[0] for index, i in enumerate(images) if i.endswith("_crop.png") and index%stride==start_index]
                images = sorted(images)
            except FileNotFoundError:
                print("{}: No such file or directory".format(path_images))
                os._exit(1)  
            l=len(images)-1
            starting_time = time.time()
            print("Starting learn image directory: {} at {}".format(line[0], time.asctime().split(" ")[3]))  
            for index,i in enumerate(images):
                depth=get_image(i+"_depthcrop.png",0, image_path=path_images)
                img=get_image(i+"_crop.png", image_path=path_images)
                mask=get_image(i+"_maskcrop.png",0, image_path=path_images)
                depth=apply_mask(mask,depth)
                img=apply_mask(mask,img)
                print("{:.2f}%".format(index*100/l)) 
                descr_path=os.path.join(path_descriptors, name, line[0])
                try:
                    with open(descr_path+"/"+i+".txt", "r") as f:
                        lines=f.readlines()
                        old_color,old_shape,old_texture=lines
                    color_descriptor=g.color_extractor.extract(img)
                    with open(descr_path+"/"+i+".txt", "w+") as f:
                        f.writelines(str(color_descriptor)+"\n")
                        f.writelines(old_shape)
                        f.writelines(old_texture)
                except:
                    continue
                #os._exit(1) #da rimuovere per fare tutte

            print("100%\nFinished learn image directory {}/{}: {} at {} in {:.2f}m ".format(number_line,number_of_object,line[0], time.asctime().split(" ")[3], (time.time()-starting_time)/60))              


                
if __name__=="__main__":
    main("classify","general",captured=True)
    #learn_knowledge()           
