import cv2
import os
import sys
from math import copysign, log10
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import random
import math
import itertools

dir_name=os.path.dirname(__file__)
image_path = os.path.join(dir_name, "object")

def get_image(filename):
    path=os.path.join(image_path,filename)
    if "maskcrop" in filename:
        im = cv2.imread(path,0)
    else: 
        im = cv2.imread(path)   
    return im

def apply_mask(mask,image):
    i=image.copy()
    i[mask == 0]=0
    return i

def calculate_descriptor(mask, rgb, depth):
    descriptors_2d=calculate_descriptors_2d(mask,depth)
    descriptors_3d=calculate_descriptors_3d(mask,rgb,depth)
    return descriptors_2d+descriptors_3d

def calculate_descriptors_3d(mask,rgb,depth):
    compactess_3d=calculate_compactess_3d(mask,rgb,depth)
    symmetry_3d=calculate_symmetry_3d(mask,rgb,depth)
    global_convexity_3d=calculate_global_convexity_3d(mask,rgb,depth)
    local_convexity_3d=calculate_local_convexity_3d(mask,rgb,depth)
    smoothness_3d=calculate_smoothness_3d(mask,rgb,depth)
    return [compactess_3d,symmetry_3d,global_convexity_3d,local_convexity_3d,smoothness_3d]

def get_roi(image):
    min_x,min_y,w,h = cv2.boundingRect(image)
    max_x=min_x+w
    max_y=min_y+h
    return image[min_y:max_y,min_x:max_x]

def calculate_descriptors_2d(mask,rgb):
    mask_roi=get_roi(mask)
    compactess_2d=calculate_compactess_2d(mask_roi)
    symmetry_2d=calculate_symmetry_2d(mask_roi)
    global_convexity_2d=calculate_global_convexity_2d(mask)
    uniqueness_2d=calculate_uniqueness_2d(mask,rgb)
    smoothness_2d=calculate_smoothness_2d(mask,rgb)
    return [compactess_2d,symmetry_2d,global_convexity_2d,uniqueness_2d,smoothness_2d]

def calculate_compactess_2d(mask):
    pixels_on = cv2.countNonZero(mask)
    pixels = mask.shape[0] * mask.shape[1]
    return pixels_on/pixels    



def calculate_symmetry_2d(mask):
    symmetries=[]
    for i in range(2):
        if i:
            mask=mask.T
        half=int(mask.shape[1]/2)
        first_half = mask[:, 0:half]
        second_half = mask[:, half+(mask.shape[1] % 2):]
        second_half = np.flip(second_half, axis=1)
        symmetry = np.sum(first_half == second_half)
        symmetries.append(symmetry/first_half.size)
        
    return float(max(symmetries))

def euclidean_distance(a,b):
    return numpy.linalg.norm(a-b)

def calculate_global_convexity_2d(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(contours[0])
    m=mask.copy()
    cv2.drawContours(m, [hull], -1, 150, 1)
    contours_pos=np.argwhere(m==150)
    points=np.argwhere(m==255)
    result=np.average(np.min(np.linalg.norm(contours_pos - points[:,None], axis=-1),axis=1))
    return float(result)
    

def calculate_uniqueness_2d(mask,rgb):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    m=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(m, contours, -1, (0,255,255), 1)
    print(contours[0])
    cv2.imshow("pippo",m)
    cv2.waitKey(0)

def calculate_smoothness_2d(mask,rgb):
    pass


def calculate_compactess_3d(mask,rgb,depth):
    pass

def calculate_symmetry_3d(mask,rgb,depth):
    pass

def calculate_global_convexity_3d(mask,rgb,depth):
    pass

def calculate_local_convexity_3d(mask,rgb,depth):
    pass

def calculate_smoothness_3d(mask,rgb,depth):
    pass

def main():
    files = os.listdir( image_path )
    files.sort()
    crop_rgb=[get_image(i) for i in files if i.endswith("_crop.png")]
    crop_masks=[get_image(i) for i in files if i.endswith("maskcrop.png")]
    crop_depth=[get_image(i) for i in files if i.endswith("depthcrop.png")]
    names=[" ".join(i.split("_")[0:-4]) for i in files if i.endswith("depthcrop.png")]
    
    depths = [ apply_mask(m,i) for m, i in zip(crop_masks,crop_depth)]
    rgbs = [ apply_mask(m,i) for m, i in zip(crop_masks,crop_rgb)]

    # cv2.imshow("Both", np.hstack((crop_rgb[1], crop_masks[1])))
    # cv2.waitKey(0)

    ref=files[0]
    descriptors_dict={}
    for name,mask,rgb,depth in zip(names,crop_masks,rgbs,depths):
        descriptors=calculate_descriptor(mask,rgb,depth)
        print("Descriptors {}: ".format(name),end='')
        for d in descriptors:
            if type(d)==float:     
                print("{:.4f}".format(d),end=' ')
        print("\n")    
        if name in descriptors_dict.keys():   
            descriptors_dict[name].append(descriptors)
        else:
            descriptors_dict[name]=[descriptors]        

        


def distance_point_point(p1,p2):
    return np.linalg.norm(p1-p2)

def distance_hull_external_poins_to_point(point, hull):
    if point_in_hull(point, hull):
        return 0.0
    return min([distance_point_point(point,v) for v in hull.vertices])

        

if __name__ == "__main__":
    main()