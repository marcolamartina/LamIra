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
        centroid_list=kmeans(pixels)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return centroid_list
     

    def classify(self,features):
        centroids={ "nero":[10,0,0],                                                            # RGB(0, 0, 0)
                    "bianco": [100,0.00526049995830391,-0.010408184525267927],                  # RGB(255, 255, 255)
                    "grigio": [53.585013452169036,0.003155620347972121,-0.006243566036245873],  # RGB(128, 128, 128)
                    "rosso":[53.23288178584245,80.10930952982204,67.22006831026425],            # RGB(255, 0, 0)
                    "verde scuro":[46.22881784262658,-51.69964732808236,49.89795230983843],     # RGB(0, 128, 0)
                    "verde chiaro":[86.54957590580997,-46.32762381560207,36.94493467106661],    # RGB(144, 238, 144)
                    "giallo":[97.13824698129729,-21.555908334832285,94.48248544644461],         # RGB(255, 255, 0)
                    "blu":[32.302586667249486,79.19666178930935,-107.86368104495168],           # RGB(0, 0, 255)
                    "magenta":[60.319933664076004,98.25421868616114,-60.84298422386232],        # RGB(255, 0, 255)
                    "ciano":[91.11652110946342,-48.079618466228716,-14.138127754846131],        # RGB(255, 165, 0)
                    "arancione":[74.93219484533535, 23.936049070113096, 78.95630717524574]      # RGB(255, 165, 0)
            }
        #labels=[(min([(l,(euclidean_distance(f,c))) for l,c in centroids.items()],key=lambda x:x[1])[0],p) for f,p in features] 
        labels=[]
        probability=0
        for l,c in centroids.items():
            p=1/(euclidean_distance(features[0][0],c)+0.0001)
            probability+=p
            labels.append((l,p))
        labels=[(l,d/probability) for l,d in labels]
        labels.sort(key=lambda x:x[1],reverse=True)            
        return labels
    
    
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
    return [cielab[0]*2.55,cielab[1]+128,cielab[2]+128]

def cv2lab_to_cielab(cv2lab):
    return [cv2lab[0]/2.55,cv2lab[1]-128,cv2lab[2]-128]

def cielab_to_rgb(lab):
    rgb=cv2.cvtColor(np.array([[cielab_to_cv2lab(lab)]]).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb.tolist()[0][0]

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
    print("Centroid list: {}".format(round_list(centroids)))



if __name__=="__main__":
    main()    