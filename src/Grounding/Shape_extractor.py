import cv2
import random
import math
import numpy as np
import os

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_shape_extractor = "/content/drive/My Drive/Tesi/Code/Grounding/"
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_shape_extractor = os.path.dirname(__file__)
    data_dir_images =os.path.join(data_dir_shape_extractor,"..","..","Media","Images") 

class Shape_extractor:
    def extract(self,image):
        humoments=self.calculate_humoments(image)
        humoments_log_trasformed=self.hu_log_trasform(humoments)
        return humoments_log_trasformed

    def classify(self,features):
        labels=["tondo", "quadrato", "esagono", "triangolo", "rettangolo", "ovale"]
        return [(l,random.random()) for l in labels]    

    def hu_log_trasform(self,humoments,default=40):
        values=[]
        for i in range(0,len(humoments)):
            if humoments[i][0]==0:
                values.append(default)
            else:        
                values.append(-1*math.copysign(1.0,humoments[i][0])*math.log10(abs(humoments[i][0])))
        ref_log=np.array(values)
        return ref_log  

    def moments_distance(self,hu1,hu2):
        return np.linalg.norm(hu1[:3]-hu2[:3])

    def get_roi(self,image,tollerance=5):
        DEPTH_VIDEO_RESOLUTION=(480,640)
        min_x,min_y,w,h = cv2.boundingRect(image)
        max_x=min_x+w
        max_y=min_y+h
        min_x=max(0,min_x-tollerance)
        min_y=max(0,min_y-tollerance)
        max_x=min(DEPTH_VIDEO_RESOLUTION[1],max_x+tollerance)
        max_y=min(DEPTH_VIDEO_RESOLUTION[0],max_y+tollerance)
        start=(min_x,min_y)
        end=(max_x,max_y)
        result = image[start[1]:end[1], start[0]:end[0]]
        percentage=max([(i,result.shape[i]/DEPTH_VIDEO_RESOLUTION[i]) for i in [0,1]],key=lambda x:x[1])
        if percentage[0]==0:
            result=cv2.resize(result,(int(result.shape[1]/percentage[1]),DEPTH_VIDEO_RESOLUTION[0]),cv2.INTER_AREA)
        else:
            result=cv2.resize(result,(DEPTH_VIDEO_RESOLUTION[1],int(result.shape[0]/percentage[1])),cv2.INTER_AREA)    
        return result           

    def calculate_humoments(self,im):
        # Threshold image
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _,im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
        
        im=self.get_roi(im)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        gradient = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, element)
        # Calculate Moments
        moment = cv2.moments(gradient)
        # Calculate Hu Moments
        hu_moments = cv2.HuMoments(moment)
        return hu_moments

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
    img=~img
    e=Shape_extractor()
    centroids=e.extract(img)
    print("Centroid list: {}".format(round_list(centroids.tolist())))


if __name__=="__main__":
    main()            