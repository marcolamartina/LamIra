import freenect
import numpy as np
import cv2
import os
import random

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_images = "/content/drive/My Drive/Tesi/Media/Images/"
except:
    if __package__:
        data_dir_images =os.path.join("./",__package__,"..","..","Media","Images")
    else:
        data_dir_images =os.path.join(os.path.dirname(__file__),"..","..","Media","Images")  

class Kinect:
    def __init__(self,verbose):
        self.verbose=verbose

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
        img = self.get_color_image()
        depth = self.get_depth_image()
        #freenect.sync_stop()
        if self.verbose:
            self.show_image(img,depth)
        return img,depth    

    def get_depth_image(self):
        return self.__pretty_depth_cv(freenect.sync_get_depth()[0])


    def get_color_image(self):
        return self.__video_cv(freenect.sync_get_video()[0])
      

    def show_image(self,img,depth):
        cv2.namedWindow('Depth')
        cv2.namedWindow('Video')
        cv2.imshow('Depth', depth)
        cv2.imshow('Video', img)

        cv2.waitKey(0)
        cv2.destroyWindow("Depth")
        cv2.destroyWindow("Video")


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

            
