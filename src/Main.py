from Controller.Controller import Controller
from Video_player.Video_player import Video_player
from Kinect.Kinect import Kinect_video_player,COLOR_VIDEO_RESOLUTION,DEPTH_VIDEO_RESOLUTION
from Intent_classification.Intent_classification import BERT_Arch
from multiprocessing import Process, Value, Lock, Array
import cv2
import os
import numpy as np
import sys


language="it-IT"
verbose=True
show_video=False
show_depth=True
show_assistent=True
play_audio=True
microphone=True
device_type="cpu"

def logic_start(close, video_id, lock, videos, default, name, image, depth, i_shape, d_shape,newstdin):
    controller=Controller(newstdin,close, verbose, show_assistent, play_audio, microphone, language,device_type,video_id, lock, videos, default, name, image, depth, i_shape, d_shape)
    controller.run()

def kinect_video_player_start(close, image, depth, i_shape, d_shape, show_video, show_depth):
    kinect_video_player=Kinect_video_player(close, image, depth, i_shape, d_shape, show_video, show_depth)
    kinect_video_player.run()

def video_player_start(close, video_id, lock, videos, default, name):
    video_player=Video_player(close, video_id, lock, videos, default, name)
    video_player.run()

def get_video_path():
    default_video="nothing.mp4"
    path = os.path.join("..","Media","Video")
    try:
        videos = os.listdir( path )
        videos=[i for i in videos if i.endswith(".mp4")]
        return videos,videos.index(default_video)
    except FileNotFoundError:
        print("{}: No such file or directory".format(path))
        os._exit(1)  

def main():
    name="LamIra"
    videos,default=get_video_path()
    video_id = Value('i',  default)
    close = Value('i',  0)
    lock = Lock()

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

    newstdin = sys.stdin.fileno()

    logic = Process(target=logic_start,args=(close, video_id, lock, videos, default, name, image, depth, i_shape, d_shape, newstdin))
    if show_assistent:
        video_player = Process(target=video_player_start,args=(close, video_id, lock, videos, default, name))
    kinect_video_player = Process(target=kinect_video_player_start,args=(close, image, depth, i_shape, d_shape, show_video, show_depth))
    
    logic.start()
    if show_assistent:
        video_player.start()
    kinect_video_player.start()
    
    logic.join()
    if show_assistent:
        video_player.join()
    kinect_video_player.join() 

if __name__=="__main__":
    main()
