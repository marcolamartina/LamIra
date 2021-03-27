from Speech_to_text.Speech_to_text import Speech_to_text
from Grounding.Grounding import Grounding
from Intent_classification.Intent_classification import BERT_Arch,Intent_classification
from Text_production.Text_production import Text_production
from Text_to_speech.Text_to_speech import Text_to_speech
from Kinect.Kinect import Kinect
import os
import random
import sys


class Controller:
    def __init__(self, newstdin=sys.stdin, close=None, verbose=False, show_assistent=True, play_audio=True, microphone=False, language="it-IT",device_type="cpu", video_id=None, lock=None, videos=None, default=None, name="LamIra", image=None, depth=None, i_shape=None, d_shape=None):
        sys.stdin = os.fdopen(newstdin)
        self.name=name
        self.close=close
        self.show_assistent=show_assistent
        self.verbose=verbose
        self.microphone=microphone
        self.intent_threshold=70
        self.intent_classification=Intent_classification(verbose,device_type,language)
        self.grounding=Grounding(verbose)
        self.text_production=Text_production(verbose)
        self.text_to_speech=Text_to_speech(verbose,language,play_audio)
        self.kinect=Kinect(verbose, image, depth, i_shape, d_shape)
        if microphone:
            self.speech_to_text=Speech_to_text(verbose,language)

        # for video player
        self.video_id=video_id
        self.lock=lock
        self.videos=videos
        self.default=default

        # welcome message
        self.say("welcome")

    def query_mode(self):
        while True:
            query=self.get_input("query")
            intents,best_intent=self.intent_classification.predict(query)
            if not self.check_intent(best_intent):
                self.say("cannot_answer")
                self.query_mode()
                return
            best_intent=best_intent[0]    
            if best_intent=="exit":
                self.say("quit")
                self.close.value=1
                return
            if best_intent=="training_mode":
                self.say("training_mode")
                self.intent_classification.set_class_type("training")
                self.training_mode()
                return      
            self.thinking()     
            #image=self.kinect.get_image_example()
            image=self.kinect.get_image()
            predictions=self.grounding.classify(image,best_intent)
            text=self.text_production.to_text_predictions(best_intent,predictions)
            self.say_text(text)    

    def training_mode(self):
        while True:
            request=self.get_input("training_request")
            intents,best_intent=self.intent_classification.predict(request)
            if not self.check_intent(best_intent):
                self.say("cannot_answer")
                self.query_mode()
                return
            best_intent=best_intent[0]    
            if best_intent=="exit":
                self.say("quit")
                self.close.value=1
                return
            if best_intent=="training_mode": # change to query mode when intent training weight are trained
                self.say("query_mode")
                self.intent_classification.set_class_type("query")
                self.query_mode()
                return      
            self.thinking()   
            #image=self.kinect.get_image_example()
            image=self.kinect.get_image()
            label=self.get_label_input(best_intent)
            self.grounding.learn(image,best_intent,label)
            text=self.text_production.to_text_subject(best_intent,label)
            self.say_text(text)

    def thinking(self):
        self.play_video("thinking")
        if self.verbose:
            print("Sto pensando...")  

    def get_input(self,request_type):
        self.say(request_type)
        self.play_video("listen")
        if self.microphone and (self.verbose or not self.show_assistent):
            print("Sto ascoltando...")
        elif not self.microphone:
            request = input("Scrivi la tua richiesta: ")
            return request
        flag=self.speech_to_text.ERROR    
        while flag!=self.speech_to_text.SUCCESS:
            flag,request=self.speech_to_text.start()
            if flag==self.speech_to_text.ERROR:
                self.say("error")
                return 
            elif flag==self.speech_to_text.UNDEFINED:
                self.say("undefined")      
        return request 

    def get_label_input(self,intent):
        request_type=intent.split("_")[0]+"_label_query"
        return self.get_input(request_type)                     

    def run(self):
        self.query_mode()

    def play_video(self, video_name):
        if self.show_assistent:
            with self.lock:
                target=video_name+".mp4"
                if target not in self.videos:
                    target=self.default 
                self.video_id.value=self.videos.index(target)

    def say_text(self, text):
        if self.show_assistent:
            target="speak.mp4" 
            with self.lock:        
                self.video_id.value=self.videos.index(target)      
        self.text_to_speech.speak(text)
        if self.show_assistent:
            with self.lock:
                self.video_id.value=self.default              


    def say(self, message):
        target=message+".mp4"
        if target not in self.videos:
            self.say_text(message)
            return
        if self.show_assistent:    
            with self.lock:        
                self.video_id.value=self.videos.index(target)      
        self.text_to_speech.speak_from_file(self.__message(message))
        if self.show_assistent:
            with self.lock:
                if message=="quit":
                    self.video_id.value=-1
                    return   
                self.video_id.value=self.default   

    def check_intent(self,best_intent):
        return best_intent[1]>self.intent_threshold      

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


