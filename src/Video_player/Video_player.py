import cv2
import os
import platform

class Video_player:
    def __init__(self, close, video_id, lock, videos, default, name):
        self.video_id=video_id
        self.lock=lock
        self.videos=videos
        self.default=default
        self.name=name
        self.close=close
        if platform.system()=="Darwin":
            self.timestep=1
        else:
            self.timestep=10
            

    def run(self):
        cv2.namedWindow(self.name)
        cv2.moveWindow(self.name,0,0)
        
        while True:
        #This is to check whether to break the first loop
            with self.lock:
                current_id=self.video_id.value
                if current_id==-1 or self.close.value==1:
                    return
                video_path=os.path.join("..","Media","Video",self.videos[current_id])
            cap = cv2.VideoCapture(video_path)
            while True:

                ret, frame = cap.read()
                # It should only show the frame when the ret is true
                if ret == True and current_id==self.video_id.value: # video not finished and no request for playing

                    cv2.imshow(self.name,frame)
                    cv2.waitKey(self.timestep)
                # elif ret == False and current_id==video_id.value: # video finished and no request for playing
                    #video_id.value=default
                    #break
                else:
                    break
            cap.release()     