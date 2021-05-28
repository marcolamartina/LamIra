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
    data_dir_images = os.path.join(data_dir_color_extractor,"..","..","Datasets","rgbd-dataset")  

class Color_extractor:
    def extract(self,image):
        image = cv2.GaussianBlur(image, (7, 7),0)
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
    rgb=cv2.cvtColor(np.array([[cielab_to_cv2lab(lab)]]).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return rgb.tolist()[0][0]

def normalize_color(color):
    color_normalized=[]
    for i,f in enumerate(color):
        if i%3==0:
            color_normalized.append(f/256)
        else:
            color_normalized.append((f+128)/256)
    return color_normalized

def print_colors(color_list,img=[]):
    print("I'm sorry, I'm not ready to plot colors yet...")
    
    '''
    # At moment pyplot cause problem used with show_assistent=True, will be fixed in future
    import matplotlib.pyplot as plt
    color_matrix=None
    for c in [color_list[i:i + 3] for i in range(0, len(color_list), 3)]:
        color=cielab_to_cv2lab(c)
        if color_matrix is not None:
            color_matrix=np.concatenate((color_matrix,np.full((20,20,3),np.array(color))),axis=0)
        else:
            color_matrix=np.full((20,20,3),np.array(color))   
    plt.figure(num="Colors")
    if len(img)>0:
        plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(color_matrix.astype(np.uint8), cv2.COLOR_LAB2RGB))
    if len(img)>0:
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    '''

def main():
    
    import random
    from Grounding import round_list

    def apply_mask(mask,image):
        i=image.copy()
        if len(image.shape)==2:
            i[mask == 0]=0
        else:
            i[mask == 0]=np.array([0,0,0])       
        return i

    try:
        images = os.listdir( data_dir_images )
        images=[i for i in images if i.endswith(".jpg")]
    except FileNotFoundError:
        print("{}: No such file or directory".format(data_dir_images))
        os._exit(1)

    for folder in sorted(os.scandir(data_dir_images),key=lambda x:x.name):
        if folder.is_dir() and "captured" not in folder.name:
            folder_name=folder.name
            for subfolder in sorted(os.scandir(os.path.join(data_dir_images,folder_name)),key=lambda x:x.name):
                if subfolder.is_dir():
                    subfolder_name=subfolder.name
                    images = os.listdir( os.path.join(data_dir_images,folder_name,subfolder_name) )
                    for i in images:
                        if i.endswith("_crop.png"):
                            filename=i
                            mask_name=i.replace("crop","maskcrop")
                            break
                    print("Image: {}".format(filename))
                    path_img = os.path.join(data_dir_images,folder_name,subfolder_name,filename)
                    path_mask= os.path.join(data_dir_images,folder_name,subfolder_name,mask_name)

                    img = cv2.imread(path_img)
                    mask = cv2.imread(path_mask,0)
                    img = apply_mask(mask,img)
                    e=Color_extractor()
                    centroids=e.extract(img)
                    print_colors(centroids,img)
                    print("Color descriptor: {}".format(round_list(centroids)))
                    #cv2.imshow("Image",img)
                    #cv2.waitKey(0)
                    #exit(0)



if __name__=="__main__":
    main()    