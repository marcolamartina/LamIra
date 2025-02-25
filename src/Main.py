from Controller.Controller import Controller
from Video_player.Video_player import Video_player
from Sensor.Kinect import Sensor_video_player as SVP_Kinect
from Sensor.Kinect import COLOR_VIDEO_RESOLUTION as COLOR_VIDEO_RESOLUTION_Kinect
from Sensor.Kinect import DEPTH_VIDEO_RESOLUTION as DEPTH_VIDEO_RESOLUTION_Kinect


from Sensor.RealSense import Sensor_video_player as SVP_RealSense
from Sensor.RealSense import COLOR_VIDEO_RESOLUTION as COLOR_VIDEO_RESOLUTION_RealSense 
from Sensor.RealSense import DEPTH_VIDEO_RESOLUTION as DEPTH_VIDEO_RESOLUTION_RealSense
from Intent_classification.Intent_classification import BERT_Arch
from multiprocessing import Process, Value, Lock, Array
import os
import numpy as np
import sys
import argparse

language="it-IT"
device_type="cpu"

verbose=False
show_video=True
show_depth=True
show_merged=True
show_assistent=True
play_audio=True
transcription=True
microphone=True
realsense=False


def logic_start(close, video_id, lock, videos, default, name, image, depth, merged, roi, i_shape, d_shape, m_shape, newstdin, calibration, realsense):
    controller=Controller(newstdin, close, verbose, show_assistent, play_audio, transcription, microphone, language,device_type,video_id, lock, videos, default, name, image, depth, merged, roi, i_shape, d_shape, m_shape, calibration, realsense)
    controller.run()

def sensor_video_player_start(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged, calibration):
    if realsense:
        video_player=SVP_RealSense(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged, calibration)
    else:
        video_player=SVP_Kinect(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged, calibration)
    video_player.run()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', action='store_true',help="activate verbose mode")
    parser.add_argument('-V','--video', action='store_false',help="disable video RGB")
    parser.add_argument('-D','--depth', action='store_false',help="disable depth")
    parser.add_argument('-M','--merged', action='store_false',help="disable image segmentated")
    parser.add_argument('-A','--assistent', action='store_false',help="disable assistent")
    parser.add_argument('-a','--audio', action='store_false',help="disable audio output")
    parser.add_argument('-t','--transcription', action='store_false',help="disable transcription of audio input")
    parser.add_argument('-m','--microphone', action='store_false',help="disable vocal input")
    parser.add_argument('-r','--realsense', action='store_true',help="use Intel RealSense")
    args = parser.parse_args()

    global verbose, show_video, show_depth, show_merged, show_assistent, play_audio, transcription, microphone, realsense
    verbose=args.verbose
    show_video=args.video
    show_depth=args.depth
    show_merged=args.merged
    show_assistent=args.assistent
    play_audio=args.audio
    transcription=args.transcription
    microphone=args.microphone
    realsense=args.realsense


    name="LAMIRA"
    videos,default=get_video_path()
    video_id = Value('i',  default)
    calibration = Value('i',  0)
    close = Value('i',  0)
    lock = Lock()

    if realsense:
        COLOR_VIDEO_RESOLUTION=COLOR_VIDEO_RESOLUTION_RealSense
        DEPTH_VIDEO_RESOLUTION=DEPTH_VIDEO_RESOLUTION_RealSense
    else:
        COLOR_VIDEO_RESOLUTION=COLOR_VIDEO_RESOLUTION_Kinect
        DEPTH_VIDEO_RESOLUTION=DEPTH_VIDEO_RESOLUTION_Kinect

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
    roi = Array('i',[-1 for _ in range(4*10)])

    newstdin = sys.stdin.fileno()

    logic = Process(target=logic_start,args=(close, video_id, lock, videos, default, name, image, depth, merged, roi, i_shape, d_shape, m_shape, newstdin, calibration, realsense))
    if show_assistent:
        video_player = Process(target=video_player_start,args=(close, video_id, lock, videos, default, name))
    sensor_video_player = Process(target=sensor_video_player_start,args=(close, image, depth, merged, roi, i_shape, d_shape, m_shape, show_video, show_depth, show_merged, calibration))

    logic.start()
    if show_assistent:
        video_player.start()
    sensor_video_player.start()
    
    logic.join()
    if show_assistent:
        video_player.join()
    sensor_video_player.join() 

if __name__=="__main__":
    main()
