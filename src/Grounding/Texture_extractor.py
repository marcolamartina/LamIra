import cv2
import random
import mahotas
import os

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_texture_extractor = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_texture_extractor = os.path.dirname(__file__)
    data_dir_images =os.path.join(data_dir_texture_extractor,"..","..","Datasets","rgbd-dataset","bell_pepper","bell_pepper_1")


class Texture_extractor:
    def extract(self,image):
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        texture_feature = mahotas.features.haralick(image,return_mean=True)
        return texture_feature.tolist()

    def classify(self,features):
        labels=["ruvido", "liscio", "intrecciato", "nido d'ape", "damascato", "feltro"]
        return [(l,random.random()) for l in labels]    

def main():
    def get_image(filename, image_path=data_dir_images):
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


    from Grounding import round_list
    try:
        files = os.listdir(data_dir_images)
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)
    e=Texture_extractor()
    files.sort()
    crop_masks=[get_image(i) for i in files if i.endswith("maskcrop.png")]
    crop_depth=[get_image(i) for i in files if i.endswith("depthcrop.png")]
    names=[" ".join(i.split("_")[0:-4]) for i in files if i.endswith("depthcrop.png")]
    depths = [ apply_mask(m,i) for m, i in zip(crop_masks,crop_depth)]
    
    name,depth=names[0],depths[0]
    descriptors=e.extract(depth)   
    print("Texture descriptors: {}".format(round_list(descriptors)))    


if __name__=="__main__":
    main()