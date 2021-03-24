from Controller.Controller import Controller
from Intent_classification.Intent_classification import BERT_Arch
from multiprocessing import Process, Value, Lock
import cv2
import os

language="it-IT"
verbose=True
device_type="cpu"

def logic_run(video_id, lock, videos, default, name):
    controller=Controller(verbose,language,device_type,video_id, lock, videos, default, name)
    controller.run()

def video_player_run(video_id, lock, videos, default, name):
    cv2.namedWindow(name)
    while True:
    #This is to check whether to break the first loop
        with lock:
            current_id=video_id.value
            if current_id==-1:
                return
            video_path=os.path.join("..","Media","Video",videos[current_id])
        isclosed=0
        cap = cv2.VideoCapture(video_path)
        while True:

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True and current_id==video_id.value: # video not finished and no request for playing

                cv2.imshow(name,frame)
                if cv2.waitKey(1) == 27:
                    # When esc is pressed isclosed is 1
                    isclosed=1
                    break
            # elif ret == False and current_id==video_id.value: # video finished and no request for playing
                #video_id.value=default
                #break
            else:
                break
        # To break the loop if it is closed manually
        if isclosed:
            break

        cap.release() 

def get_video_path():
    path = os.path.join("..","Media","Video")
    try:
        videos = os.listdir( path )
        videos=[i for i in videos if i.endswith(".mp4")]
        return videos,videos.index("nothing.mp4")
    except FileNotFoundError:
        print("{}: No such file or directory".format(path))
        os._exit(1)  

def main():
    name="LamIra"
    videos,default=get_video_path()
    video_id = Value('i',  default)
    lock = Lock()

    logic = Process(target=logic_run,args=(video_id, lock, videos, default, name))
    video_player = Process(target=video_player_run,args=(video_id, lock, videos, default, name))

    logic.start()
    video_player.start()
    
    logic.join()
    video_player.join() 

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