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
show_video=True
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

'''
centroids={ "nero":[10,0,0],                                                            # RGB(0, 0, 0)
            "bianco": [100,0.00526049995830391,-0.010408184525267927],                  # RGB(255, 255, 255)
            "grigio": [53.585013452169036,0.003155620347972121,-0.006243566036245873],  # RGB(128, 128, 128)
            "rosso":[53.23288178584245,80.10930952982204,67.22006831026425],            # RGB(255, 0, 0)
            "verde scuro":[46.22881784262658,-51.69964732808236,49.89795230983843],     # RGB(0, 128, 0)
            "verde chiaro":[86.54957590580997,-46.32762381560207,36.94493467106661],    # RGB(144, 238, 144)
            "giallo":[97.13824698129729,-21.555908334832285,94.48248544644461],         # RGB(255, 255, 0)
            "blu":[32.302586667249486,79.19666178930935,-107.86368104495168],           # RGB(0, 0, 255)
            "magenta":[60.319933664076004,98.25421868616114,-60.84298422386232],        # RGB(255, 0, 255)
            "ciano":[91.11652110946342,-48.079618466228716,-14.138127754846131],        # RGB(255, 165, 0)
            "arancione":[74.93219484533535, 23.936049070113096, 78.95630717524574]      # RGB(255, 165, 0)
            }
'''