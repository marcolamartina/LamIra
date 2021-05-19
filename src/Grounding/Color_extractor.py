import cv2
import math
import os
import numpy as np
if __package__:
    from Grounding.Clustering import kmeans, euclidean_distance
else:
    from Clustering import kmeans, euclidean_distance

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_color_extractor = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_color_extractor = os.path.dirname(__file__)
    data_dir_images =os.path.join(data_dir_color_extractor,"..","..","Media","Images")    

class Color_extractor:
    def extract(self,image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        rows,cols,channels = lab.shape
        pixels=[cv2lab_to_cielab(lab[i,j]) for j in range(cols) for i in range(rows) if image[i,j].tolist()!=[0,0,0]]
        centroid_list=kmeans(pixels, n_clusters=3, dimensions=False)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        centroid_list_flatten=[item for sublist in centroid_list for item in sublist]
        return centroid_list_flatten
    
    
    def abs_cartesian_to_polar(self, p):
        """Accepts a point given in cartesian coordinates
        and converts it to polar coordinates 

        :p:   Cartesian coordinates in the (x, y) format
        :returns:  Cartesian coordinates in the (Rho, Theta) format

        """
        x,y = p
        angle = math.degrees(math.atan2(y,x))
        return (math.hypot(x,y), angle) 

def cielab_to_cv2lab(cielab):
    return [cielab[0]*2.56,cielab[1]+128,cielab[2]+128]

def cv2lab_to_cielab(cv2lab):
    return [cv2lab[0]/2.56,cv2lab[1]-128,cv2lab[2]-128]

def cielab_to_rgb(lab):
    rgb=cv2.cvtColor(np.array([[cielab_to_cv2lab(lab)]]).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb.tolist()[0][0]

def normalize_color(color):
    color_normalized=[]
    for i,f in enumerate(color):
        if i%3==0:
            color_normalized.append(f/256)
        else:
            color_normalized.append((f+128)/256)
    return color_normalized
   

def main():
    import random
    from Grounding import round_list
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
    e=Color_extractor()
    centroids=e.extract(img)
    print("Color descriptor: {}".format(round_list(centroids)))



if __name__=="__main__":
    main()    