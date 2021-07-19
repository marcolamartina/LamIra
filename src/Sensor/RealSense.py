import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import random
import time
if __package__:
    from Sensor.Image_processing import Image_processing
else:
    from Image_processing import Image_processing

COLOR_VIDEO_RESOLUTION=(480,640,3)
DEPTH_VIDEO_RESOLUTION=(480,640)


data_dir_images =os.path.join(os.path.dirname(__file__),"..","..","Media","Images")  

class RealSense:
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


    
class Sensor_video_player:
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
        self.pipeline = self.__init_sensor__()

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
        while True:
            try:
                if self.close.value==1: 
                    return
                if self.calibration.value==1:
                    self.image_processing.depth_base=self.get_depth_image()
                    self.calibration.value=0
                    continue
                i = array_to_image(self.i_arr,self.i_shape)
                d = array_to_image(self.d_arr,self.d_shape)
                m = array_to_image(self.m_arr,self.m_shape)            

                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    return

                # Convert images to numpy arrays
                depth = np.asanyarray(depth_frame.get_data())
                image = np.asanyarray(color_frame.get_data())
                
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
        depth_frame=None
        while not depth_frame:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def __pretty_depth(self,depth):
        """Converts depth into a 'nicer' format for display

        This is abstracted to allow for experimentation with normalization

        Args:
            depth: A numpy array with 2 bytes per pixel

        Returns:
            A numpy array that has been processed with unspecified datatype
        """
        #np.clip(depth, 0, 2**10 - 1, depth)
        #depth >>= 2
        #depth = depth.astype(np.uint8)
        return depth


    def get_color_image(self):
        color_frame=None
        while not color_frame:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image     


    def __init_sensor__(self):
        pipeline=None
        try:
            # Configure depth and color streams
            pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
        except:
            print("ERRORE: Non è stato rilevato alcun sensore Intel RealSense")
            self.close.value=1
            os._exit(1)
        return pipeline

    def __del__(self):
        if self.pipeline:
            self.pipeline.stop()
        

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

    sensor_video_player=Sensor_video_player(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged)
    #sensor_video_player.run()
    depth, frame=sensor_video_player.get_depth_image()
    
    np.save(data_dir_images+"depth_realsense.npy", depth)    
    np.save(data_dir_images+"frame_realsense.npy", frame)

if __name__=="__main__":
    main()
