#!/usr/bin/env python
import freenect
import cv2
import os
import numpy as np

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


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


def get_mask(depth,start=1,slices_selected=4,slices=5):
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


if __name__=="__main__":
    from_kinect=False


    for i in range(1):
        # Defining starting element
        path=os.path.dirname(__file__)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        depth_file=path+"/test/depth_"+str(i)

        # Take depth image
        if from_kinect:
            dep_original=get_depth()
        else:
            dep_original=cv2.imread(depth_file+"_starting.png")
            v=get_video()

        # Remove background by mask
        depth_masked=get_mask(dep_original)

        # Filtering depth

        d_median = cv2.medianBlur(depth_masked, 5)
        bord = 10
        d_median_cropped = d_median[bord:d_median.shape[0]-bord, bord:d_median.shape[1]-bord]

        #d_edge = cv2.Canny(depth_masked,55,65,apertureSize = 5)
        #d_edge_2 = cv2.morphologyEx(d_edge, cv2.MORPH_GRADIENT, element)
        #d_edge_3 = cv2.medianBlur(d_edge_2, 5)
        #d_median_cropped = crop_image(depth_masked, 50)

        # Filtering Video
        if v:
            v_grey = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            v_edge=cv2.Canny(v_grey,70,90, apertureSize = 3)
            rgb_file=path+"/test/rgb_"+str(i)

        # Clear rows with uniform values (First floor removal)
        for i in range(d_median_cropped.shape[0]-1,-1, -1):
            row = d_median_cropped[i, :]
            if (np.max(row) - np.min(row)) < 8:
                d_median_cropped[i, :]=0

        # Save interediet file
        d_first_attempt = d_median_cropped.copy()
        d_final2 = get_mask(d_first_attempt, slices_selected=15, slices=16)

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
            #d_adpt_thres = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            #cv2.imwrite(depth_file+str(i)+"_roi.png", th2)

        # Saving Files
        cv2.imwrite(depth_file+"_starting.png", dep_original)
        cv2.imwrite(depth_file+"_without_background.png", depth_masked)
        cv2.imwrite(depth_file+"_final.png", d_first_attempt)
        cv2.imwrite(depth_file+"_final_2nd_get_mask.png", d_final2)
        cv2.imwrite(depth_file+"_floor_removal_by_column.png", d_median_cropped)
        #cv2.imwrite(depth_file+"_adpt_thres.png", d_adpt_thres)
        #np.save(depth_file+".npy", depth_masked) # save

        '''
        cv2.imwrite(depth_file+"_edge.png", d_edge)
        cv2.imwrite(depth_file+"_edge2.png", d_edge_2)
        cv2.imwrite(depth_file+"_edge3.png", d_edge_3)
        cv2.imwrite(rgb_file+".png", v)
        np.save(rgb_file+".npy", v) # save
        cv2.imwrite(rgb_file+".png", v)
        cv2.imwrite(rgb_file+"_edge.png", v_edge)
        '''