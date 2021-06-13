from scipy.sparse.construct import kron
from Speech_to_text.Speech_to_text import Speech_to_text
from Grounding.Grounding import Grounding
from Intent_classification.Intent_classification import BERT_Arch,Intent_classification
from Text_production.Text_production import Text_production
from Text_to_speech.Text_to_speech import Text_to_speech
from Kinect.Kinect import Kinect
import os
import random
import sys
import traceback
import numpy as np


class Controller:
    def __init__(self, newstdin=sys.stdin, close=None, verbose=False, show_assistent=True, play_audio=True, microphone=False, language="it-IT",device_type="cpu", video_id=None, lock=None, videos=None, default=None, name="LamIra", image=None, depth=None, merged=None, roi=None, i_shape=None, d_shape=None, m_shape=None, calibration=None):
        sys.stdin = os.fdopen(newstdin)
        self.name=name
        self.close=close
        self.show_assistent=show_assistent
        self.verbose=verbose
        self.microphone=microphone
        self.intent_threshold=60
        self.intent_classification=Intent_classification(verbose,device_type,language)
        self.grounding=Grounding(verbose)
        self.text_production=Text_production(verbose)
        self.text_to_speech=Text_to_speech(verbose,language,play_audio)
        self.kinect=Kinect(verbose, image, depth, merged, roi, i_shape, d_shape, m_shape)
        if microphone:
            self.speech_to_text=Speech_to_text(verbose,language)

        # for video player
        self.video_id=video_id
        self.lock=lock
        self.videos=videos
        self.default=default
        self.calibration=calibration
        self.spaces=["general","color","shape","texture"]
        self.confirm_query={"unsure_answer":"È corretto?", "dubious_answer":"Cosa è in realtà?", "cannot_answer":"Potresti dirmi come si chiama?"}
        self.negative_sentences=["no","negativo"]
        self.cancel_sentences=["annulla","esci","niente","nulla"]

        # welcome message
        self.say("welcome")

    def query_mode(self):
        while True:
            try:
                if self.close.value==1:
                    self.say("quit")
                    return
                query=self.get_input("query")
                if "calibrazione" in query:
                    self.calibration.value=1
                    self.say_text("Calibrazione completata")
                    continue
                if "reset" in query:
                    self.grounding.reset_knowledge()
                    self.say_text("Reset delle conoscenze completato")
                    continue    
                intents,best_intent=self.intent_classification.predict(query)
                if not self.check_intent(best_intent):
                    self.say("cannot_answer")
                    self.query_mode()
                    return
                best_intent=best_intent[0]    
                if best_intent=="exit":
                    self.say("quit")
                    return
                if best_intent=="training_mode":
                    self.say("training_mode")
                    self.intent_classification.set_class_type("training")
                    self.training_mode()
                    return      
                self.thinking()     
                rois=self.kinect.get_image_roi()
                roi=max(rois,key=lambda x:np.count_nonzero(x[1]))
                predictions,description,features=self.grounding.classify(roi,best_intent)
                text,prediction_type=self.text_production.to_text_predictions(best_intent,predictions,description)
                self.say_text(text)
                if prediction_type!="sure_answer":
                    confirm_response=self.get_input("confirm_label",self.confirm_query[prediction_type])
                    label_confirmed,label_correct=self.verify_prediction(confirm_response)
                    if label_confirmed==2:
                        continue
                    if label_correct:
                        self.grounding.learn_features(best_intent,label_correct,features)

                
            except ValueError:
                self.say_text("Non è stato rilevato nessun oggetto")     
            except:
                self.log(sys.exc_info()[1])
                self.say("error")
                continue
                

    def training_mode(self):
        while True:
            try:
                if self.close.value==1:
                    self.say("quit")
                    return
                request=self.get_input("training_request")
                if "calibrazione" in request:
                    self.calibration.value=1
                    self.say_text("Calibrazione completata")
                    continue
                if "reset" in request:
                    self.grounding.reset_knowledge()
                    self.say_text("Reset delle conoscenze completato")
                    continue    
                intents,best_intent=self.intent_classification.predict(request)
                if not self.check_intent(best_intent):
                    self.say("cannot_answer")
                    self.training_mode()
                    return
                best_intent=best_intent[0]    
                if best_intent=="exit":
                    self.say("quit")
                    return
                if best_intent=="query_mode":
                    self.say("query_mode")
                    self.intent_classification.set_class_type("query")
                    self.query_mode()
                    return      
                self.thinking()   
                rois=self.kinect.get_image_roi()
                roi=max(rois,key=lambda x:np.count_nonzero(x[1]))  

                label_confirmed=0
                while not label_confirmed:
                    labels=self.get_label_input(best_intent)
                    confirm_response=self.get_input("confirm_label","Vuoi confermare {}?".format(self.concatenate_labels(labels)))
                    label_confirmed=self.verify_confirm(confirm_response)
                if label_confirmed==2:
                    continue

                for i,l in enumerate(labels):
                    if l:    
                        self.grounding.learn(roi,self.spaces[i]+"_training",l)
                text=self.text_production.to_text_subject(labels)
                self.say_text(text)
            
            except ValueError:
                self.say_text("Non è stato rilevato nessun oggetto")     
            except:
                self.log(sys.exc_info()[1])
                self.say("error")
                continue  
    
    def concatenate_labels(self,labels):
        labels=[i for i in labels if i]
        if len(labels)<=1:
            return labels[0]
        return " e ".join([", ".join(labels[:-1]),labels[-1]])

    def verify_confirm(self,confirm_response):
        """Accepts a text and return an integer that can be:
        - 0 if the response is negative
        - 1 if the response is affermative
        - 2 for cancel the operation 

        """
        for n in confirm_response.lower().split(" "):
            if n in self.cancel_sentences:
                return 2
            elif n in self.negative_sentences:
                return 0
        return 1

    def verify_prediction(self,confirm_response):
        """Accepts a text and return an integer that can be:
        - 0 if the response is negative
        - 1 if the response is affermative
        - 2 for cancel the operation 

        and if the response is negative, return also the correct label
        """
        l=confirm_response.lower().split(" ")
        for n in l:
            if n in self.cancel_sentences:
                return 2,None
            elif n in self.negative_sentences:
                return 0,l[-1]
        return 1,None  

        
               

    def thinking(self):
        self.play_video("thinking")
        if self.verbose:
            print("Sto pensando...")  

    def get_input(self,request_type,text=None):
        self.say(request_type,text)
        self.play_video("listen")
        if self.microphone and (self.verbose or not self.show_assistent):
            print("Sto ascoltando...")
        elif not self.microphone:
            request = input("Puoi scrivere: ")
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
        user_input=self.get_input(request_type)
        input_list=user_input.split(" ")
        result=[input_list[0]]
        for l in ["colore","forma","tessitura"]:
            if l in input_list:
                result.append(input_list[input_list.index(l)+1])
        return result                      

    def run(self):
        self.query_mode()

    def play_video(self, video_name):
        if self.show_assistent:
            with self.lock:
                target=video_name+".mp4"
                if target not in self.videos:
                    target=self.default 
                self.video_id.value=self.videos.index(target)

    def check_intent(self,best_intent):
        return best_intent[1]>self.intent_threshold  

    def say_text(self, text):
        if self.show_assistent:
            target="speak.mp4" 
            with self.lock:        
                self.video_id.value=self.videos.index(target)      
        self.text_to_speech.speak(text)
        if self.show_assistent:
            with self.lock:
                self.video_id.value=self.default              

    def say(self, video, text=None):
        if video in ["training_request","query"]:
            return
        if self.show_assistent:
            target=video+".mp4"
            if target not in self.videos:
                target="speak.mp4"    
            with self.lock:        
                self.video_id.value=self.videos.index(target)
        if text==None:
            self.text_to_speech.speak_from_file(self.__message(video))
        else:
            self.text_to_speech.speak(text)
        with self.lock:
            if video=="quit":
                if self.show_assistent:
                    self.video_id.value=-1
                self.close.value=1
                return   
            self.video_id.value=self.default               

    def __message(self, message_type):
        if message_type==None:
            return None
        path = os.path.join("..","Media/Audio/", message_type.capitalize())
        try:
            files = os.listdir( path )
            files=[i for i in files if i.endswith(".mp3")]
            return message_type.capitalize()+"/"+random.choice(files)
        except FileNotFoundError:
            self.error("{}: No such file or directory".format(path),"C'è stato un problema di configurazione")

    def error(self,text="Si è verificato un errore",audio=None):
        if not audio:
            audio=text
        self.verbose=False   
        self.say_text(audio)
        print(text)
        os._exit(1)

    def log(self,stacktrace):
        print(stacktrace, file = sys.stderr)

