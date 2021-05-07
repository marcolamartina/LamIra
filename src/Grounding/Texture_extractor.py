import cv2
import random
import mahotas  
import os
import numpy as np

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_texture_extractor = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_texture_extractor = os.path.dirname(__file__)
    data_dir_images = os.path.join(data_dir_texture_extractor,"..","..","Datasets","rgbd-dataset")
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))
    data_dir_images = os.path.join(data_dir_images,random.choice([f.name for f in os.scandir(data_dir_images) if f.is_dir() and not f.name.startswith("_")]))


class Texture_extractor:
    def extract(self,image):
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        texture_feature = mahotas.features.haralick(image)
        return np.mean(texture_feature,axis=0).tolist()

    def classify(self,features):
        labels=["ruvido", "liscio", "intrecciato", "nido d'ape", "damascato", "feltro"]
        return [(l,random.random()) for l in labels]    

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
    e=Texture_extractor()
    descriptors=e.extract(depth)   
    print("Texture descriptors: {}".format(round_list(descriptors)))    


if __name__=="__main__":
    main()