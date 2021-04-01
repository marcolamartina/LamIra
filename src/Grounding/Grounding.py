if __package__:
    from Grounding.Color_extractor import Color_extractor
    from Grounding.Shape_extractor import Shape_extractor
    from Grounding.Texture_extractor import Texture_extractor
else:
    from Color_extractor import Color_extractor
    from Shape_extractor import Shape_extractor
    from Texture_extractor import Texture_extractor    
import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_grounding = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_grounding = os.path.dirname(__file__)
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
    def classify(self, scan, intent):
        image,depth,merged=scan
        features={}
        features['color']=self.color_extractor.extract(merged)
        features['shape']=self.shape_extractor.extract(image)
        features['texture']=self.texture_extractor.extract(image)
        features['general']=sorted([(i, random.random()) for i in ["palla da tennis", "uovo", "banana", "pera", "anguria", "anguilla"]],key=lambda x:x[1])
         
        if self.verbose:
            print("Intent: {}".format(intent))
            print("Color: {}\nColor features: {}\n".format(round_list(features['color'][0]),round_list(features['color'][1])))
            self.print_colors(features['color'][1])
            print("Shape: {}\nShape features: {}\n".format(round_list(features['shape'][0]),round_list(features['shape'][1]))) 
            print("Texture: {}\nTexture features: {}\n".format(round_list(features['texture'][0]),round_list(features['texture'][1])))         
        return features[intent[:-6]][0]


    def learn(self, scan, intent, label):
        image,depth,merged=scan
        return

    def print_colors(self,color_list):
        color_matrix=None
        for c in color_list:
            color=self.color_extractor.cielab_to_cv2lab(c[0])
            if color_matrix is not None:
                color_matrix=np.concatenate((color_matrix,np.full((20,20,3),np.array(color))),axis=0)
            else:
                color_matrix=np.full((20,20,3),np.array(color))   
        plt.imshow(cv2.cvtColor(color_matrix.astype(np.uint8), cv2.COLOR_LAB2RGB))
        plt.show()    

def main():
    import cv2
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
    result=g.classify((img,depth,img),"color_query")
    print(round_list(result))


if __name__=="__main__":
    main()            


