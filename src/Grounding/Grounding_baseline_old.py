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



''' 
TO BE ACQUIRED

SENZA ADDESTRAMENTO
.--------------------------------------------------------------------------------------------------------------------.
| OGGETTO       |  POSIZIONE 1                    |    POSIZIONE 2                   |    POSIZIONE 3                |
|--------------------------------------------------------------------------------------------------------------------|
| Aglio 1       | 95.09267013399148, -1.3681241184767294, -2.3244005641748915, 95.09267013399148, -1.3681241184767294, -2.3244005641748915, 65.35998233215548, -1.3074204946996484, -7.289752650176686, 0.6703877790834313, 0.8343125734430082, 0.20067640335750928, 2.196675143502039, 0.7923380097309888, 0.35118497999384424, -2.6132840353130184, -0.0056123248283569535, 0.48496594154730177, 1.7294831532499466, 0.4523264767297567, 4.999623331103596, 0.9580652448566519, 0.3308228870601847, 0.7762232374771356, 0.7474724045447728, 1.2955158630679413, 3.189158006992921, 3.733989726658287, 0.002197246173956103, 2.1607867955113145, -0.6401459183345197, 0.9849152768966902                                |                                  |                               |  
| Aglio 2       |                                 |                                  |                               |  
| Aglio 3       |                                 |                                  |                               |  
| Spazzolino 1  | 46.485800638686136, 3.521897810218976, -25.63868613138687, 78.71690124045803, -2.3473282442747996, -13.22519083969466, 27.148437500000025, 2.267175572519083, -4.629770992366387, 0.558041958041958, 0.5340909090909091, 0.10302719779160135, 2.1188829835026572, 0.6587500122656635, 0.11044982698961937, -2.4333766294043495, -0.0055688359952481455, 0.47979871358376835, 1.5751491616744593, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0                                |                                  |                               |  
| Spazzolino 2  | 34.42402906378602, 1.1049382716049387, -2.3786008230452715, 37.373218201754405, 3.9144736842105132, -30.100877192982473, 61.19540662650601, -1.4216867469879617, -14.619277108433739, 0.20053199349785725, 0.6519402985074627, 0.09228238990384177, 2.3198869757426146, 0.6072602597106166, 0.11013716419121825, -2.235966771640067, -0.005929838608272888, 0.468110113580501, 1.4025620175064513, 0.6758730130851908, 1.5903907221186824, 0.9303462330241659, 0.06748844481396436, 0.8482340260185064, 0.21839510980024798, 0.2605431832669895, 2.063639378255023, 2.526661564861765, 0.002854440133672476, 1.5370429008034177, -0.5655794729814563, 0.9286022222236044                                |                                  |                               |  
| Spazzolino 3  |                                 |                                  |                               |  
| Cipolla 1     | 90.24803755961844, -2.449920508744038, -3.588235294117645, 67.0229473431498, -1.3854033290653014, -5.016645326504478, 37.0754665798611, -1.229166666666667, 0.7934027777777728, 0.7515151515151515, 0.8750733137829912, 0.2352995567569597, 2.3214845970320277, 0.5347173674867586, 0.4629333333333333, -2.524296839571263, -0.004300492846917566, 0.4840953151032503, 1.7489559689309937, 0.30976390197848497, 2.186561648513044, 0.9742479860000401, 0.18860486214003788, 0.6364335056589286, 0.7043485064847391, 0.7447013967889825, 4.716855037873139, 5.957710921968878, 0.0014047587020134076, 2.862153410229845, -0.5747559619028677, 0.9958638474150826                                |                                  |                               |  
| Cipolla 2     | 89.71862549800801, -2.3394422310756955, -3.555378486055776, 66.7797208281054, -1.779171894604767, -4.380175658720198, 36.805010098743296, -0.22082585278276845, -0.6247755834829394, 0.7677836566725456, 0.8936678614097969, 0.23425629867698164, 2.349678911276115, 0.5927754435294338, 0.47699050401753107, -2.535854456390913, -0.004738559903868553, 0.48376137059603963, 1.7440575465288402, 0.309002239168193, 2.353744857834279, 0.9728167814750567, 0.19851388710519455, 0.6340366179003728, 0.7257963768400748, 0.7832585536600705, 4.732704659833567, 5.980549115869654, 0.0013911481763891399, 2.877423760416077, -0.5668409864728734, 0.995544589954432                                |                                  |                               |  
| Cipolla 3     | 90.37154579010375, -2.1540303272146857, -4.008778930566643, 67.52203002610966, -1.7023498694516979, -5.257180156657961, 37.38447319778189, -0.8650646950092408, -0.5397412199630329, 0.7637017070979335, 0.8807060255629945, 0.22857732720533155, 2.265946648805031, 0.7668900527931594, 0.4785137924563708, -2.5362811149535043, -0.004575350247400025, 0.4871581170613621, 1.743148292644788, 0.3048841334506362, 2.249790929519898, 0.9726829108939723, 0.18458691314884318, 0.6276057616145437, 0.6998543592870151, 0.728258904032503, 4.783713887146225, 6.051043246415938, 0.001368628293995806, 2.90970419421288, -0.5697568756353749, 0.9959267224895756                                |                                  |                               |  
| Lattina 1     |                                 |                                  |                               |  
| Lattina 2     |                                 |                                  |                               |  
| Lattina 3     |                                 |                                  |                               |  
| Limone 1      | 83.53623466257672, -17.37072743207712, 81.17879053461868, 46.69856385601579, -2.124260355029584, 10.46548323471395, 57.59028817365272, -12.137724550898202, 56.80838323353294, 0.6990891346779441, 0.873342175066313, 0.22332762714488694, 2.230402536675183, 0.7933901009645112, 0.40326515293676113, -2.34348431044622, -0.0042124050543750294, 0.4841988935412376, 1.7231323309689364, 0.37191220244031, 2.6439724511130533, 0.9659271163843257, 0.21663858952897372, 0.6879793555647484, 0.6971237160478188, 0.8517835622996209, 4.124766285378981, 5.170331228530117, 0.0017605331578200785, 2.453224482047838, -0.5677111776570412, 0.9915780531557725                                |                                  |                               |  
| Limone 2      |                                 |                                  |                               |  
| Limone 3      |                                 |                                  |                               |  
| Palla 1       | 31.693707233037742, 19.55242868157286, 37.992675404780336, 31.693707233037742, 19.55242868157286, 37.992675404780336, 19.9395622591522, 16.189306358381504, 25.557321772639746, 0.764, 0.9458181818181818, 0.2659394062583112, 2.262291981059443, 0.5333299962925866, 0.5836111111111111, -2.273576023149809, -0.003965733536882188, 0.4711590082122612, 1.8803709075510326, 0.1969052242249379, 0.8147256497032556, 0.9642265195239087, 0.10950010088522812, 0.6578804059870601, 0.6906894107695828, 0.43016650306299636, 4.840154021610501, 6.180112347431107, 0.0026380322943532738, 2.289372480398165, -0.5338019050339549, 0.9942772913153759                                |                                  |                               |  
| Palla 2       |                                 |                                  |                               |  
| Palla 3       |                                 |                                  |                               |  
| Piatto 1      |                                 |                                  |                               |  
| Piatto 2      |                                 |                                  |                               |  
| Piatto 3      |                                 |                                  |                               |  
| Tastiera 1    | 2.498721810745085, -1.0217112535927266, 0.03749723634754665, 2.498721810745085, -1.0217112535927266, 0.03749723634754665, 9.787407063197033, -0.3742255266418807, -8.097273853779413, 0.6046101694915255, 0.712180790960452, 0.12110771587299538, 2.3591466817751985, 0.5674044270808517, 1.4679835390946503, -2.0978866225254555, -0.020111948888960663, 0.4888574833933601, 1.833381041187454, 0.20612098769630594, 0.41290978196871486, 0.8516277329390385, 0.009801521772753103, 0.6605775960785563, 0.1019425559714459, 0.036298271725035545, 4.260834644074278, 5.783709622303453, 0.0023336187399720246, 2.5681126310688134, -0.4268831704617797, 0.977392046170917                                |                                  |                               |  
| Tastiera 2    |                                 |                                  |                               |  
| Tastiera 3    |                                 |                                  |                               |  
.--------------------------------------------------------------------------------------------------------------------.


'''