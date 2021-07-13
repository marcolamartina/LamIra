import freenect
import numpy as np
import cv2
import os
import sys
import signal
import random
import time
if __package__:
    from Sensor.Image_processing import Image_processing
else:
    from Image_processing import Image_processing
from contextlib import contextmanager

accept_color =  {   
                "OFF":0, 
                "GREEN":1,
                "RED":2, 
                "YELLOW":3, 
                "BLINK_GREEN":4, 
                "BLINK_RED_YELLOW":5
                }
accept_positions =  {
                    "DOWN":-27,
                    "CENTER":0,
                    "UP":27
                    }


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


data_dir_images =os.path.join(os.path.dirname(__file__),"..","..","Media","Images")  


class Kinect:
    def __init__(self, verbose, image, depth, merged, roi, i_shape, d_shape, m_shape):
        self.verbose=verbose
        self.image=image 
        self.depth=depth
        self.merged=merged
        self.i_shape=i_shape
        self.d_shape=d_shape
        self.m_shape=m_shape
        self.roi=roi

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

    def get_image_roi(self):
        #i = array_to_image(self.image,self.i_shape)
        d = array_to_image(self.depth,self.d_shape)
        m = array_to_image(self.merged,self.m_shape)
        result=[]
        for k in range(0,len(self.roi),4):
            if self.roi[k]==-1:
                break
            start=(self.roi[k],self.roi[k+1])
            end=(self.roi[k+2],self.roi[k+3])
            #i_roi = i[start[1]:end[1], start[0]:end[0]]
            d_roi = d[start[1]:end[1], start[0]:end[0]]
            m_roi = m[start[1]:end[1], start[0]:end[0]]
            _,mask=cv2.threshold(cv2.cvtColor(m_roi, cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)
            d_m_roi = d_roi*mask
            result.append((m_roi,d_m_roi))
        return result


    
class Video_player:
    def __init__(self, close, i_arr, d_arr, m_arr, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged, calibration):
        self.i_arr=i_arr
        self.d_arr=d_arr
        self.m_arr=m_arr
        self.i_shape=i_shape
        self.d_shape=d_shape
        self.m_shape=m_shape
        self.roi=roi
        self.show_video=show_video
        self.show_depth=show_depth
        self.show_merged=show_merged
        self.close=close
        self.ctx, self.dev= self.__init_kinect__()        
        self.set_led('GREEN')
        #self.set_tilt_degs(-22)
        freenect.close_device(self.dev)
        freenect.shutdown(self.ctx)
        self.calibration=calibration
        self.image_processing=Image_processing(self.get_depth_image())                

    def run(self):
        from screeninfo import get_monitors
        screen = get_monitors()
        if len(screen)>1:
            start_x=screen[1].width
        else:
            start_x=0
        window_y=1920-COLOR_VIDEO_RESOLUTION[0]
        #start_im=np.zeros((COLOR_VIDEO_RESOLUTION[0], COLOR_VIDEO_RESOLUTION[1]))
        if self.show_video:
            cv2.namedWindow('Video')
            #cv2.imshow('Video', start_im)
            cv2.moveWindow('Video',start_x , window_y)
        if self.show_depth:
            cv2.namedWindow('Depth')
            #cv2.imshow('Depth', start_im)
            cv2.moveWindow('Depth',start_x+COLOR_VIDEO_RESOLUTION[1],window_y)
        if self.show_merged:
            cv2.namedWindow('Merged')
            #cv2.imshow('Merged', start_im)
            cv2.moveWindow('Merged',start_x+2*COLOR_VIDEO_RESOLUTION[1],window_y)   
        with stderr_redirected(to=os.devnull):
            while True:
                try:
                    if self.close.value==1: 
                        freenect.sync_stop()
                        return
                    if self.calibration.value==1:
                        self.image_processing.depth_base=self.get_depth_image()
                        self.calibration.value=0
                        continue
                    i = array_to_image(self.i_arr,self.i_shape)
                    d = array_to_image(self.d_arr,self.d_shape)
                    m = array_to_image(self.m_arr,self.m_shape)
                    depth=self.get_depth_image()
                    image=self.get_color_image()
                    if depth is None or image is None:
                        return
                    image=self.image_processing.homography(image,depth)
                    merged,mask=self.image_processing.segmentation(image,depth)
                    d[...]=depth
                    i[...]=image
                    m[...]=merged
                    merged=self.get_roi(mask,merged)
                    if self.show_depth:
                        cv2.imshow('Depth', depth)
                        cv2.moveWindow('Depth',start_x+COLOR_VIDEO_RESOLUTION[1],window_y)
                    if self.show_video:   
                        cv2.imshow('Video', image)
                        cv2.moveWindow('Video',start_x , window_y)
                    if self.show_merged:   
                        cv2.imshow('Merged', merged)    
                        cv2.moveWindow('Merged',start_x+2*COLOR_VIDEO_RESOLUTION[1],window_y)   
                    if (self.show_depth or self.show_video or self.show_merged):
                        cv2.waitKey(1)
                                
                except KeyboardInterrupt:
                    freenect.sync_stop()
                    self.__del__() 
                    os._exit(1)   

    def get_roi(self,mask,output,tollerance=10):
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
        roi=[]
        for cnt in contours:
            min_x,min_y,w,h = cv2.boundingRect(cnt)
            max_x=min_x+w
            max_y=min_y+h
            min_x=max(0,min_x-tollerance)
            min_y=max(0,min_y-tollerance)
            max_x=min(COLOR_VIDEO_RESOLUTION[1],max_x+tollerance)
            max_y=min(COLOR_VIDEO_RESOLUTION[0],max_y+tollerance)
            if max_x - min_x >50 or max_y-min_y>50:
                start_point=(min_x,min_y)
                end_point=(max_x,max_y)
                roi+=[min_x,min_y,max_x,max_y]
                cv2.rectangle(output, start_point, end_point, (0, 255, 255), 2)
        self.set_roi(roi)
        return output
                        

    def set_roi(self,points):
        limit=min(len(self.roi),len(points))
        for i in range(limit):
            self.roi[i]=points[i]
        for i in range(limit,len(self.roi)):
            self.roi[i]=-1     
        

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
            self.close.value=1
            os._exit(1)
        return ctx, dev

    def __del__(self):
        self.ctx, self.dev= self.__init_kinect__() 
        self.set_led('RED')
        self.set_tilt_degs('DOWN')
        time.sleep(2)
        self.set_led('OFF')
        freenect.close_device(self.dev)
        freenect.shutdown(self.ctx)

    def set_led(self, color):
        color=color.upper()
        if color not in accept_color.keys():
            print("Color must be in corret form. \ne.g. " + str(accept_color.keys()))
        else:
            freenect.set_led(self.dev, accept_color[color])

    def set_tilt_degs(self, degree):
        if degree in accept_positions.keys():
            degree = accept_positions[degree]

        if not accept_positions["DOWN"]<=degree<=accept_positions["UP"] :
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
    roi = Array('i',[0,0,0,0]) 

    close = Value('i',  0)

    video_player=Video_player(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged)
    video_player.run()


if __name__=="__main__":
    main()
