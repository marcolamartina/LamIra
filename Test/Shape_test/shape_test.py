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


def calculate_descriptors_2d(mask,depth):
    

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