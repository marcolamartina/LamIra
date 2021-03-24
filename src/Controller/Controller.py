from Speech_to_text.Speech_to_text import Speech_to_text
from Grounding.Grounding import Grounding
from Intent_classification.Intent_classification import BERT_Arch,Intent_classification
from Text_production.Text_production import Text_production
from Text_to_speech.Text_to_speech import Text_to_speech
from Kinect.Kinect import Kinect
import os
import random

class Controller:
    def __init__(self,verbose=False,language="it-IT",device_type="cpu",video_id=None, lock=None, videos=None, default=None, name="LamIra"):
        self.name=name
        self.speech_to_text=Speech_to_text(verbose,language)
        self.intent_classification=Intent_classification(verbose,device_type,language)
        self.grounding=Grounding(verbose)
        self.text_production=Text_production(verbose)
        self.text_to_speech=Text_to_speech(verbose,language)
        self.kinect=Kinect(verbose)

        # for video player
        self.video_id=video_id
        self.lock=lock
        self.videos=videos
        self.default=default

        # welcome message
        self.say("welcome")
        

    def run(self):
        while True:
            flag=self.speech_to_text.ERROR
            self.say("query")
            self.play_video("listen")
            while flag!=self.speech_to_text.SUCCESS:
                flag,query=self.speech_to_text.start()
                if flag==self.speech_to_text.ERROR:
                    self.say("error")
                    return
                elif flag==self.speech_to_text.QUIT:
                    self.say("quit")
                    return  
                elif flag==self.speech_to_text.UNDEFINED:
                    self.say("undefined")
            intents,best_intent=self.intent_classification.predict(query)
            if not self.check_intent(best_intent):
                self.say("cannot_answer")
                self.run()
                return
            self.play_video("thinking")    
            best_intent=best_intent[0]    
            #image=self.kinect.get_image_example()
            image=self.kinect.get_image()
            predictions=self.grounding.classify(image,best_intent)
            text=self.text_production.to_text(best_intent,predictions)
            self.say_text(text)

    def play_video(self, video_name):
        with self.lock:
            target=video_name+".mp4"
            if target not in self.videos:
                target=self.default 
            self.video_id.value=self.videos.index(target)

    def say_text(self, text):
        target="speak.mp4" 
        with self.lock:        
            self.video_id.value=self.videos.index(target)      

        self.text_to_speech.speak(text)
        with self.lock:
            self.video_id.value=self.default              


    def say(self, message):
        target=message+".mp4"
        if target not in self.videos:
            target="speak.mp4" 
        with self.lock:        
            self.video_id.value=self.videos.index(target)      

        self.text_to_speech.speak_from_file(self.__message(message))
        with self.lock:
            if message=="quit":
                self.video_id.value=-1
                return   
            self.video_id.value=self.default   

    def check_intent(self,best_intent):
        return best_intent[1]>70      

    def __message(self, message_type):
        if message_type==None:
            return None
        path = os.path.join("..","Media/Audio/", message_type.capitalize())
        try:
            files = os.listdir( path )
            files=[i for i in files if i.endswith(".mp3")]
            return message_type.capitalize()+"/"+random.choice(files)
        except FileNotFoundError:
            print("{}: No such file or directory".format(path))
            os._exit(1)



