import freenect
import numpy as np
import cv2
import os
import sys
import random
from contextlib import contextmanager

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
    def __init__(self, verbose, image, depth, i_shape, d_shape):
        self.verbose=verbose
        self.image=image 
        self.depth=depth
        self.i_shape=i_shape
        self.d_shape=d_shape

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


    
class Kinect_video_player:
    def __init__(self, close, i_arr, d_arr, i_shape, d_shape, show_video, show_depth):
        self.i_arr=i_arr
        self.d_arr=d_arr
        self.i_shape=i_shape
        self.d_shape=d_shape
        self.show_video=show_video
        self.show_depth=show_depth
        self.close=close


    def run(self):
        window_y=450
        if self.show_video:
            cv2.namedWindow('Video')
            cv2.moveWindow('Video',0,window_y)
        if self.show_depth:
            cv2.namedWindow('Depth')
            cv2.moveWindow('Depth',COLOR_VIDEO_RESOLUTION[1],window_y)
        with stderr_redirected(to=os.devnull):
            while True:
                if self.close.value==1:
                    return
                i = array_to_image(self.i_arr,self.i_shape)
                d = array_to_image(self.d_arr,self.d_shape)
                depth=self.get_depth_image()
                image=self.get_color_image()
                if self.show_depth:
                    cv2.imshow('Depth', depth)
                if self.show_video:   
                    cv2.imshow('Video', image)
                if (self.show_depth or self.show_video) and cv2.waitKey(10) == 27:   
                    break
                d[...]=depth
                i[...]=image

    def get_depth_image(self):
        return self.__pretty_depth_cv(freenect.sync_get_depth()[0])


    def get_color_image(self):
        return self.__video_cv(freenect.sync_get_video()[0])        


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
                    
def array_to_image(image,shape):
    i = np.frombuffer(image.get_obj(), dtype=np.uint8)
    i.shape = shape
    return i

def main():
    from multiprocessing import Process, Value, Lock, Array
    show_video=True
    show_depth=True
    i_arr = np.zeros(COLOR_VIDEO_RESOLUTION,dtype=int)
    d_arr = np.zeros(DEPTH_VIDEO_RESOLUTION,dtype=int)

    i_shape = i_arr.shape
    i_size = i_arr.size
    i_arr.shape = i_size

    d_shape = d_arr.shape
    d_size = d_arr.size
    d_arr.shape = d_size

    image = Array('B', i_arr)
    depth = Array('B', d_arr)

    kinect_video_player=Kinect_video_player(image, depth, i_shape, d_shape, show_video, show_depth)
    kinect_video_player.run() 

if __name__=="__main__":
    main()
