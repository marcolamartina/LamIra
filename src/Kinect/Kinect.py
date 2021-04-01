import freenect
import numpy as np
import cv2
import os
import sys
import signal
import random
import time
if __package__:
    from Kinect.Image_processing import Image_processing
else:
    from Image_processing import Image_processing
from contextlib import contextmanager

OFF=0
GREEN=1
RED=2
YELLOW=3
BLINK_GREEN=4
BLINK_RED_YELLOW=5
CENTER=0
UP=30
DOWN=-30

COLOR_VIDEO_RESOLUTION=(480,640,3)
DEPTH_VIDEO_RESOLUTION=(480,640)

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stderr_redirected(to=os.devnull, stderr=None):
    if stderr is None:
       stderr = sys.stderr

    stderr_fd = fileno(stderr)
    # copy stderr_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied: 
        stderr.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stderr_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stderr_fd)  # $ exec > to
        try:
            yield stderr # allow code to be run with the redirected stderr
        finally:
            # restore stderr to its previous value
            #NOTE: dup2 makes stderr_fd inheritable unconditionally
            stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)  # $ exec >&copied



try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    data_dir_images =os.path.join(os.path.dirname(__file__),"..","..","Media","Images")  


class Kinect:
    def __init__(self, verbose, image, depth, merged, i_shape, d_shape, m_shape):
        self.verbose=verbose
        self.image=image 
        self.depth=depth
        self.merged=merged
        self.i_shape=i_shape
        self.d_shape=d_shape
        self.m_shape=m_shape

    def get_image_example(self):
        try:
            images = os.listdir( data_dir_images )
            images=[i for i in images if i.endswith(".jpg")]
        except FileNotFoundError:
            print("{}: No such file or directory".format(data_dir_images))
            os._exit(1)
        image=random.choice(images) 
        if self.verbose:
            print("Image: {}".format(image))   
        path = os.path.join(data_dir_images,image)
        img = cv2.imread(path)
        depth = None
        return img,depth

    def get_image(self):
        i = array_to_image(self.image,self.i_shape)
        d = array_to_image(self.depth,self.d_shape)
        return i,d

    def get_merged(self):
        m = array_to_image(self.merged,self.m_shape)
        return m    

    
class Kinect_video_player:
    def __init__(self, close, i_arr, d_arr, m_arr, i_shape, d_shape, m_shape, show_video, show_depth, show_merged):
        self.i_arr=i_arr
        self.d_arr=d_arr
        self.m_arr=m_arr
        self.i_shape=i_shape
        self.d_shape=d_shape
        self.m_shape=m_shape
        self.show_video=show_video
        self.show_depth=show_depth
        self.show_merged=show_merged
        self.image_processing=Image_processing()
        self.close=close

        self.ctx, self.dev= self.__init_kinect__()        
        self.set_led('GREEN')
        self.set_tilt_degs(CENTER)
        freenect.close_device(self.dev)
        freenect.shutdown(self.ctx)


    def run(self):
        window_y=450
        if self.show_video:
            cv2.namedWindow('Video')
            cv2.moveWindow('Video',0 ,window_y)
        if self.show_depth:
            cv2.namedWindow('Depth')
            cv2.moveWindow('Depth',COLOR_VIDEO_RESOLUTION[1],window_y)
        if self.show_merged:
            cv2.namedWindow('Merged')
            cv2.moveWindow('Merged',COLOR_VIDEO_RESOLUTION[1],0)      
        with stderr_redirected(to=os.devnull):
            while True:
                try:
                    if self.close.value==1: 
                        freenect.sync_stop()
                        return
                    i = array_to_image(self.i_arr,self.i_shape)
                    d = array_to_image(self.d_arr,self.d_shape)
                    m = array_to_image(self.m_arr,self.m_shape)
                    depth=self.get_depth_image()
                    image=self.get_color_image()
                    if depth is None or image is None:
                        return
                    image=self.image_processing.homography(image,depth)
                    merged=self.image_processing.segmentation(image,depth)
                    d[...]=depth
                    i[...]=image
                    m[...]=merged
                    roi_start,roi_end=self.get_roi(merged)
                    merged = cv2.rectangle(merged, roi_start, roi_end, (0, 255, 255), 2)
                    if self.show_depth:
                        cv2.imshow('Depth', depth)
                    if self.show_video:   
                        cv2.imshow('Video', image)
                    if self.show_merged:   
                        cv2.imshow('Merged', merged)    
                    if (self.show_depth or self.show_video or self.show_merged):
                        cv2.waitKey(1)
                                
                except KeyboardInterrupt:
                    freenect.sync_stop()
                    self.__del__() 
                    os._exit(1)   


    def get_roi(self,image,tollerance=5):
        min_x=COLOR_VIDEO_RESOLUTION[0]
        min_y=COLOR_VIDEO_RESOLUTION[1]
        max_x=0
        max_y=0
        for i in np.ndindex(image.shape[:2]):
            if image[i].tolist()!=[0,0,0]:
                if i[0]<min_x:
                    min_x=i[0]
                if i[1]<min_y:
                    min_y=i[1]
                if i[0]>max_x:
                    max_x=i[0]
                if i[1]>max_y:
                    max_y=i[1]
        for i in range(COLOR_VIDEO_RESOLUTION[0]):
            


        min_x=max(0,min_x-tollerance)
        min_y=max(0,min_y-tollerance)
        max_x=min(COLOR_VIDEO_RESOLUTION[0],max_x+tollerance)
        max_y=min(COLOR_VIDEO_RESOLUTION[1],max_y+tollerance)
        start_point=(min_y,min_x)
        end_point=(max_y,max_x)
        return start_point,end_point                     


    def get_depth_image(self):
        f=freenect.sync_get_depth()
        if f==None:
            self.close.value=1
            return None
        return self.__pretty_depth_cv(f[0])


    def get_color_image(self):
        f=freenect.sync_get_video()
        if f==None:
            self.close.value=1
            return None
        return self.__video_cv(f[0])        


    def __pretty_depth(self,depth):
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


    def __pretty_depth_cv(self,depth):
        """Converts depth into a 'nicer' format for display

        This is abstracted to allow for experimentation with normalization

        Args:
            depth: A numpy array with 2 bytes per pixel

        Returns:
            A numpy array with unspecified datatype
        """
        return self.__pretty_depth(depth)


    def __video_cv(self,video):
        """Converts video into a BGR format for display

        This is abstracted out to allow for experimentation

        Args:
            video: A numpy array with 1 byte per pixel, 3 channels RGB

        Returns:
            A numpy array with with 1 byte per pixel, 3 channels BGR
        """
        return video[:, :, ::-1]  # RGB -> BGR

    def __init_kinect__(self):
        ctx = freenect.init()
        dev = freenect.open_device(ctx, freenect.num_devices(ctx) - 1)

        if not dev:
            freenect.error_open_device()
            os._exit(1)
        return ctx, dev

    def __del__(self):
        self.ctx, self.dev= self.__init_kinect__() 
        self.set_led('RED')
        self.set_tilt_degs(DOWN)
        time.sleep(2)
        self.set_led('OFF')
        freenect.close_device(self.dev)
        freenect.shutdown(self.ctx)

    def set_led(self, color):
        color=color.upper()
        accept_color=["OFF", "GREEN", "RED", "YELLOW", "BLINK_GREEN", "BLINK_RED_YELLOW"]
        if color not in accept_color:
            print("Color must be in corret form. \ne.g. " + str(accept_color))
        else:
            freenect.set_led(self.dev, globals()[color])

    def set_tilt_degs(self, degree):
        if type(degree)==str:
            degree = int(globals()[degree])

        if abs(degree) > UP:
            print("Degree angle must be beetween -30 and 30")
        else:
            freenect.set_tilt_degs(self.dev, degree)
                    
def array_to_image(image,shape):
    i = np.frombuffer(image.get_obj(), dtype=np.uint8)
    i.shape = shape
    return i

def main():
    from multiprocessing import Process, Value, Lock, Array
    show_video=True
    show_depth=True
    show_merged=True

    i_arr = np.zeros(COLOR_VIDEO_RESOLUTION,dtype=int)
    d_arr = np.zeros(DEPTH_VIDEO_RESOLUTION,dtype=int)
    m_arr = np.zeros(COLOR_VIDEO_RESOLUTION,dtype=int)

    i_shape = i_arr.shape
    i_size = i_arr.size
    i_arr.shape = i_size

    d_shape = d_arr.shape
    d_size = d_arr.size
    d_arr.shape = d_size

    m_shape = m_arr.shape
    m_size = m_arr.size
    m_arr.shape = m_size

    image = Array('B', i_arr)
    depth = Array('B', d_arr)
    merged = Array('B', m_arr)

    close = Value('i',  0)

    kinect_video_player=Kinect_video_player(close, image, depth, merged, i_shape, d_shape, m_shape, show_video, show_depth, show_merged)
    kinect_video_player.run() 

if __name__=="__main__":
    main()
