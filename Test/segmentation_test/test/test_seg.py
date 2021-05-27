import cv2
import os
import numpy as np

def padding(image, border_size=10):
    return cv2.copyMakeBorder(image, border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)

def crop(image, border_size=10):
    return image[border_size:image.shape[0]-border_size, border_size:image.shape[1]-border_size]


path=os.path.dirname(__file__)

#load base
base_rgb = cv2.imread(path+"/rgb.png")
base_depth = cv2.imread(path+"/depth.png",0)

#load img with object
obj_1_rgb = cv2.imread(path+"/rgb_obj_1.png")
obj_1_depth = cv2.imread(path+"/depth_obj_1.png",0)

obj_2_rgb = cv2.imread(path+"/rgb_obj_2.png")
obj_2_depth = cv2.imread(path+"/depth_obj_2.png",0)

obj_3_rgb = cv2.imread(path+"/rgb_obj_3.png")
obj_3_depth = cv2.imread(path+"/depth_obj_3.png",0)

#subtract
subs = []
subs.append(obj_1_depth-base_depth)
subs.append(obj_2_depth-base_depth)
subs.append(obj_3_depth-base_depth)
for i, sub in enumerate(subs):
    sub[sub==255] = 0
    sub[sub<10] = 0
    sub[sub>10] = 255
    cv2.imwrite(path+"/segmented_"+str(i)+".png", sub)
    sub = crop(sub)
    sub = padding(sub)
    subs[i] = cv2.medianBlur(sub, 7)
    #cv2.imshow("depth sub"+str(i), sub)
    cv2.imwrite(path+"/segmented_median_"+str(i)+".png", subs[i])

#cv2.imshow("Rgb sub", obj_1_rgb-base_rgb)
for i, sub in enumerate(subs):
    cv2.imshow("depth sub"+str(i), sub)

cv2.waitKey(0)
