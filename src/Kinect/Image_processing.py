import numpy as np
import cv2

class Image_processing:
    def __init__(self, depth_base):
        self.depth_base = depth_base

    def homography(self,color,depth):
        h=self.get_homography()
        return self.apply_homography(color,h,depth.shape)

    def segmentation(self, color, depth):
        #mask=self.get_mask(depth) #Adaptive method for find mask
        mask=self.get_mask_by_sub(depth) #Use a image calibration to get mask
        merged=self.merge(color,mask)
        return merged,mask

    def get_homography(self):
        h = np.array([[ 1.14747786e+00, 3.84008813e-02, -2.07838484e+01],
                      [ 5.69715874e-03, 1.15317345e+00, -4.77633142e+01],
                      [ 7.64673839e-06, 1.08779788e-04, 1.00000000e+00]])             
        return h

    def calculate_homography(self):
        # corners of depth
        point_depth=[[ 436 , 15 ], [ 101 , 287 ], [442, 475], [622,259], [44, 325], [119,363], [129, 344], [187, 101], [188, 257], [410, 254],[406, 102]]
        pts_depth = np.array(point_depth)

        # corners of image
        points_image=[[ 393 , 55 ], [ 98 , 293 ], [401, 476], [571, 272], [43, 339], [120,364],[122, 352], [175, 132],[184, 271],[383, 264],[379, 126]]
        pts_image = np.array(points_image)

        # Calculate Homography
        h, status = cv2.findHomography(pts_image,pts_depth)

        return h

    def apply_homography(self,image,h,shape):
        im_out = cv2.warpPerspective(image, h, (shape[1],shape[0]))
        return im_out

    def select_slices(self,depth,start=1,slices_selected=4,slices=5):
        minimum=np.min(depth[depth>0])
        result=depth.copy()
        result[depth==255]=0
        maximum=np.max(result)
        step=(maximum-minimum)/slices
        threshold_1=(step*(start-1))+minimum
        threshold_2=threshold_1+(step*slices_selected)
        result[depth < threshold_1] = 0
        #result[depth >= threshold_1] = 255
        result[depth > threshold_2] = 0
        return result

    def get_point(self,image,border_size=10,tolerance=10):
        return (image.shape[0]-border_size-tolerance,image.shape[1]-border_size-tolerance)           
   
    def padding(self, image, border_size=10):
        return cv2.copyMakeBorder(image, border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)

    def crop(self, image, border_size=10):
        return image[border_size:image.shape[0]-border_size, border_size:image.shape[1]-border_size]

    def border_padding(self, image, border_size=(20,20,20,20)):
        '''Pad image
        :image: input image
        :border_size: (left,right,upper,bottom)

        :returns: image with border padded 
        '''
        im=np.copy(image)
        v=0
        im[:, 0:border_size[0]]=v #Left
        im[:, -border_size[1]:]=v #Right
        im[0:border_size[2], :]=v #Upper
        im[-border_size[3]:, :]=v #Bottom
        return im

    def remove_floor(self, original, output, border_size=10, point=None, tolerance=10):
        if not point:
            point = self.get_point(original,border_size)
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

    def get_mask_by_sub(self, depth, threshold=5):
        d=depth.astype(np.float32)
        b=self.depth_base.astype(np.float32)
        mask=np.abs(d-b)
        mask[mask<=threshold] = 0
        mask[mask>threshold] = 255
        mask=mask.astype(np.uint8)
        mask = self.border_padding(mask)
        return cv2.medianBlur(mask, 3)

    def get_mask(self,depth):
        # Remove background
        depth_no_background=self.select_slices(depth)

        # Filtering depth
        depth_no_background_median = cv2.medianBlur(depth_no_background, 5)

        # Cropping image
        border_size=10
        depth_no_background_median_cropped = self.crop(depth_no_background_median,border_size=border_size)
        # Padding image for restore dimensions        
        depth_no_background_median_cropped = self.padding(depth_no_background_median_cropped,border_size=border_size)

        # Sobel
        depth_sobel = cv2.Sobel(depth,cv2.CV_64F,0,1,ksize=5)
        abs_sobel64f = np.absolute(depth_sobel)
        depth_sobel = np.uint8(abs_sobel64f)

        # Blurring Sobel image
        kernel_sobel_size=9
        depth_sobel_blurred = cv2.blur(depth_sobel, (kernel_sobel_size,kernel_sobel_size))

        depth_no_floor=self.remove_floor(depth_sobel_blurred,depth_no_background_median_cropped,border_size)
        depth_no_floor_blurred=cv2.medianBlur(depth_no_floor, 15)
        depth_no_floor_blurred[depth_no_floor_blurred>0]=255
        return depth_no_floor_blurred

    def scale(self,x,maximum,minimum):
        if x==0:
            return 0  
        a=(x-minimum)/(maximum-minimum)
        return 1-a

    def change_brightness(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img        

    def reshape(self,image):
        original_size=image.shape
        image=image[30:,10:590,:]
        resized = cv2.resize(image, (original_size[1],original_size[0]), interpolation=cv2.INTER_AREA)
        return resized

    def merge(self,image,depth):
        i=image.copy()
        i[depth < 255]=0
        return i           


def main(mod):
    import freenect
    import os

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

    def mouseRGB(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
            colors = color[y,x]
            print("Color value: ",colors)
            print("Coordinates color: [{},{}]".format(x,y))


    def mouseDepth(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
            color = depth[y,x]
            print("Depth value: ",color)
            print("Coordinates depth: [",x,",",y,"]")     

    def processing_image():
        ip=Image_processing()
        path=os.path.dirname(__file__)
        depth_file=path+"/test_images/depth.npy"
        color_file=path+"/test_images/image.npy"

        depth = np.load(depth_file) # load
        color = np.load(color_file) # load

        color=ip.homography(color,depth)
        merged,mask=ip.segmentation(color,depth)
        globals()["color"]=color
        globals()["depth"]=depth
        globals()["merged"]=merged
        while(1):
            cv2.imshow('Depth', depth)
            cv2.setMouseCallback('Depth',mouseDepth)
            cv2.imshow('Video', color)
            cv2.setMouseCallback('Video',mouseRGB)
            cv2.imshow('Merged', merged)
            cv2.setMouseCallback('Merged',mouseRGB)
            if cv2.waitKey(0)== 27:
                break
        #if esc pressed, finish.
        cv2.destroyAllWindows()

    def processing_video():
        ip=Image_processing()
        #Do until esc pressed
        while(1):
            depth=get_depth()
            color=get_video()
            color=ip.homography(color,depth)
            merged,mask=ip.segmentation(color,depth)
            globals()["color"]=color
            globals()["depth"]=depth
            globals()["merged"]=merged
            cv2.imshow('Depth', depth)
            cv2.setMouseCallback('Depth',mouseDepth)
            cv2.imshow('Video', color)
            cv2.setMouseCallback('Video',mouseRGB)
            cv2.imshow('Merged', merged)
            cv2.setMouseCallback('Merged',mouseRGB)
            if cv2.waitKey(10)== 27:
                break
        #if esc pressed, finish.
        cv2.destroyAllWindows()

    if mod=="video":
        processing_video()
    elif mod=="image":
        processing_image()   

if __name__=="__main__":
    main("video")