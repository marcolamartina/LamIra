#!/usr/bin/env python3
import os
import numpy as np
import cv2
import freenect

def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array that has been processed with unspecified datatype
    """
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth


def pretty_depth_cv(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array with unspecified datatype
    """
    return pretty_depth(depth)


def video_cv(video):
    """Converts video into a BGR format for display

    This is abstracted out to allow for experimentation

    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB

    Returns:
        A numpy array with with 1 byte per pixel, 3 channels BGR
    """
    return video[:, :, ::-1]  # RGB -> BGR

def get_depth():
    return pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return video_cv(freenect.sync_get_video()[0])

print_images=True

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        #print("Red: ",colorsR)
        #print("Green: ",colorsG)
        #print("Blue: ",colorsB)
        print("Color value: ",colors)
        print("Coordinates color: [{},{}]".format(x,y))


def mouse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        color = depth[y,x]
        print("Depth value: ",color)
        print("Coordinates depth: [",x,",",y,"]")

def elaborate(depth,slices_selected=2,slices=3):
    coefficient=slices/slices_selected
    minimum=np.min(depth)
    d=depth.copy()
    maximum=np.max(d)
    threshold=((maximum-minimum)/coefficient)+minimum
    d[depth > threshold] = 0
    d[depth <= threshold] = 255
    return d

def reshape(image):
    original_size=image.shape
    image=image[30:,10:590,:]
    resized = cv2.resize(image, (original_size[1],original_size[0]), interpolation=cv2.INTER_AREA)
    return resized

def merge(image,depth):
    i=image.copy()
    i[depth < 255]=0
    return i

def check_depth(x, threshold):
    if x>=threshold:
        return 1
    return 0    

def scale(x,maximum,minimum):
    if x==0:
        return 0  
    a=(x-minimum)/(maximum-minimum)
    return 1-a 

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def calculate_homography():
    import cv2
    import numpy as np

    # Four corners of depth
    point_depth=[[ 436 , 15 ], [ 101 , 287 ], [442, 475], [622,259], [44, 325], [119,363], [129, 344], [187, 101], [188, 257], [410, 254],[406, 102]]
    pts_depth = np.array(point_depth)

    # Four corners of image
    points_image=[[ 393 , 55 ], [ 98 , 293 ], [401, 476], [571, 272], [43, 339], [120,364],[122, 352], [175, 132],[184, 271],[383, 264],[379, 126]]
    pts_image = np.array(points_image)

    # Calculate Homography
    h, status = cv2.findHomography(pts_image,pts_depth)

    return h

def apply_homography(image,h,shape):
    im_out = cv2.warpPerspective(image, h, (shape[1],shape[0]))
    return im_out


def segment_image():
    path=os.path.dirname(__file__)
    depth_file=path+"/depth_chair.npy"
    image_file=path+"/image_chair.npy"

    depth = np.load(depth_file) # load
    image = np.load(image_file) # load
    h=calculate_homography()
    image=apply_homography(image,h,depth.shape)
    depth=elaborate(depth)
    merged=merge(image,depth)
    #merged=change_brightness(merged)

    if print_images:
        #Do until esc pressed
        while(1):
            globals()["image"]=image
            globals()["depth"]=depth
            globals()["merged"]=merged
            cv2.imshow('Depth', depth)
            cv2.setMouseCallback('Depth',mouse)
            cv2.imshow('Video', image)
            cv2.setMouseCallback('Video',mouseRGB)
            cv2.imshow('Merged', merged)
            cv2.setMouseCallback('Merged',mouseRGB)
            if cv2.waitKey(0)== 27:
                break
        #if esc pressed, finish.
        cv2.destroyAllWindows()



def segment_video():
    h=calculate_homography()


    #Do until esc pressed
    while(1):
        depth=get_depth()
        image=get_video()
        image=apply_homography(image,h,depth.shape)
        depth=elaborate(depth)
        #merged=merge(image,depth)
        #merged=change_brightness(merged)
        cv2.imshow('Depth', depth)
        cv2.setMouseCallback('Depth',mouse)
        cv2.imshow('Video', image)
        cv2.setMouseCallback('Video',mouseRGB)
        #cv2.imshow('Merged', merged)
        #cv2.setMouseCallback('Merged',mouseRGB)
        if cv2.waitKey(10)== 27:
            break
    #if esc pressed, finish.
    cv2.destroyAllWindows()

if __name__=="__main__":
    segment_image()