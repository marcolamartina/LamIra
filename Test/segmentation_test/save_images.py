#!/usr/bin/env python
import freenect
import cv2
import os
import numpy as np
import time

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

def get_contours(grey):
    contours, hierarchy = cv2.findContours(grey,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    roi=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        g=grey.copy()
        roi.append(g[y:y+h,x:x+w])
        
        cv2.rectangle(grey,(x,y),(x+w,y+h),(200,0,0),2)
    
    return grey, roi  


def select_slices(depth,start=1,slices_selected=4,slices=5):
    minimum=np.min(depth[depth>0])
    depth_masked=depth.copy()
    depth_masked[depth==255]=0
    maximum=np.max(depth_masked)
    step=(maximum-minimum)/slices
    threshold_1=(step*(start-1))+minimum
    threshold_2=threshold_1+(step*slices_selected)
    depth_masked[depth < threshold_1] = 0
    #depth_masked[depth >= threshold_1] = 255
    depth_masked[depth > threshold_2] = 0
    return depth_masked

def crop_image(image, zoom):
    image_h, image_w = image.shape
    h_desired = int(image_h*zoom/100)
    w_desired = int(image_w*zoom/100)

    zoomed=cv2.resize(image, (h_desired ,w_desired), interpolation=cv2.INTER_AREA)
    cv2.imshow("pippo", zoomed)
    cv2.waitKey(0)

    delta_h=int(h_desired-image_h/2)
    delta_w=int(w_desired-image_w/2)
    return  image[delta_h:image_h+delta_h, delta_w:image_w+delta_w]

def padding(image, border_size=10):
    return cv2.copyMakeBorder(image, border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)

def crop(image, border_size=10):
    return image[border_size:image.shape[0]-border_size, border_size:image.shape[1]-border_size]

def get_point(image,border_size=10,tolerance=10):
    return (image.shape[0]-border_size-tolerance,image.shape[1]-border_size-tolerance)

def removing_bottom_floor(image):
    # Clear rows with uniform values (First floor removal)
    for i in range(image.shape[0]-1,-1, -1):
        row = image[i, :]
        if (np.max(row) - np.min(row)) < 8:
            image[i, :]=0
    return image

def get_neighbours(point, max_dimensions):
    l=[(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    return {(point[0]+x,point[1]+y) for x,y in l if 0<=point[0]+x<max_dimensions[0] and 0<=point[1]+y<max_dimensions[1]}            
   
    
def removing_floor_region_growing(image, output, border_size=10, point=None, tolerance=25):
    if not point:
        point = get_point(image,border_size)
    result=output.copy()    
    value_ref=int(image[point[0],point[1]] )   
    neighbours={point}
    visited=set()
    while len(neighbours)>0:
        p=neighbours.pop()
        n=set()
        n={x for x in get_neighbours(p, image.shape) if x not in visited and abs(int(image[x[0],x[1]])-value_ref)<tolerance}    
        neighbours.update(n)
        result[p[0],p[1]]=0
        visited.add(p)   
    return result

def removing_floor_connected_components(original, output, border_size=10, point=None, tolerance=8):
    if not point:
        point = get_point(original,border_size)
    image=original.copy()
    result=output.copy()    
    value_ref=int(image[point[0],point[1]] )    
    image[image<value_ref-tolerance]=0
    image[image>value_ref+tolerance]=0
    image[image>0]=1
    _, labels = cv2.connectedComponents(image)
    label=labels[point[0],point[1]]
    result[labels==label]=0
    return result

def live_mode():
    while True:
        # Original depth
        dep_original=get_depth()
 
        # Remove background
        depth_no_background=select_slices(dep_original)

        # Filtering depth
        depth_no_background_median = cv2.medianBlur(depth_no_background, 5)

        # Cropping image
        border_size=10
        depth_no_background_median_cropped = crop(depth_no_background_median,border_size=border_size)
        # Padding image for restore dimensions        
        depth_no_background_median_cropped = padding(depth_no_background_median_cropped,border_size=border_size)

        # Sobel
        depth_sobel = cv2.Sobel(dep_original,cv2.CV_64F,0,1,ksize=5)
        abs_sobel64f = np.absolute(depth_sobel)
        depth_sobel = np.uint8(abs_sobel64f)

        # Blurring Sobel image
        kernel_size=9
        depth_sobel_blurred = cv2.blur(depth_sobel, (kernel_size,kernel_size))

        depth_without_floor=removing_floor_connected_components(depth_sobel_blurred,depth_no_background_median_cropped,border_size)
        depth_without_floor_blurred=cv2.medianBlur(depth_without_floor, 15)
        cv2.imshow('Depth Without Backgroud and Floor', depth_without_floor_blurred)
        cv2.imshow('Depth Sobel', depth_sobel_blurred)

        if cv2.waitKey(1)==27:
            break

def batch_mode():
    from_kinect=False
    
    for i in range(1):
        # Defining starting element
        path=os.path.dirname(__file__)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        depth_file=path+"/test/depth_"+str(i)

        # Take depth image
        if from_kinect:
            cv2.namedWindow('Depth')
            cv2.namedWindow('Video')
            dep_original=get_depth()
            v=get_video()
        else:
            dep_original=cv2.imread(depth_file+"_starting.png", 0)
            v=None

        
        # Remove background by mask
        depth_masked=select_slices(dep_original)       

        # Filtering depth
        d_median = cv2.medianBlur(depth_masked, 5)

        border_size=10
        # Cropping image
        d_median_cropped = crop(d_median,border_size=border_size)
        # Padding image for restore dimensions        
        d_median_cropped = padding(d_median_cropped,border_size=border_size)
        
        

        #d_edge = cv2.Canny(depth_masked,55,65,apertureSize = 5)
        #d_edge_2 = cv2.morphologyEx(d_edge, cv2.MORPH_GRADIENT, element)
        #d_edge_3 = cv2.medianBlur(d_edge_2, 5)
        #d_median_cropped = crop_image(depth_masked, 50)

        # Filtering Video
        if v:
            v_grey = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            v_edge=cv2.Canny(v_grey,70,90, apertureSize = 3)
            rgb_file=path+"/test/rgb_"+str(i)


        # Save interediet file
        d_first_attempt = d_median_cropped.copy()
        d_final2 = select_slices(d_first_attempt, slices_selected=15, slices=16)


        depth_sobel = cv2.Sobel(dep_original,cv2.CV_64F,0,1,ksize=5)
        abs_sobel64f = np.absolute(depth_sobel)
        depth_sobel = np.uint8(abs_sobel64f)

        kernel_size=9
        depth_test = cv2.blur(depth_sobel, (kernel_size,kernel_size))
        cv2.imshow("sobel", depth_sobel)
        cv2.imshow("pippo", depth_test)

        image_without_floor=removing_floor_region_growing(depth_test,d_median_cropped,border_size)
        image_without_floor=cv2.medianBlur(image_without_floor, 15)  
        cv2.imshow("not floor", image_without_floor)
        cv2.waitKey(0)


        # Clear clomuns with floor
        for i in range(d_median_cropped.shape[1]-1,-1, -1):
            column=d_median_cropped[:, i]
            obj=False
            for j,value in enumerate(column):
                if j == len(column)-1:
                    break
                succ = column[j+1]
                if value > 0 and value < succ:
                    obj=True
                    break
                    
            if not obj:
                d_median_cropped[:, i]=0

        d_median_cropped = cv2.medianBlur(d_median_cropped, 5)
        #d_median_cropped = cv2.morphologyEx(d_median_cropped, cv2.MORPH_GRADIENT, element)

        # Extracting ROI
        d_median_cropped, rois = get_contours(d_median_cropped)
        for i, roi in enumerate(rois):
            ret2,th2 = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # Saving Files
        cv2.imwrite(depth_file+"_starting.png", dep_original)
        cv2.imwrite(depth_file+"_without_background.png", depth_masked)
        cv2.imwrite(depth_file+"_final.png", d_first_attempt)
        cv2.imwrite(depth_file+"_final_2nd_select_slices.png", d_final2)
        cv2.imwrite(depth_file+"_floor_removal_by_column.png", d_median_cropped)


if __name__=="__main__":
    live_mode()         