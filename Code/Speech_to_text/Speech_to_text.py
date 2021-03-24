import random
import time
import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone,language):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio, language=language)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response



class Speech_to_text:
    def __init__(self,verbose=False,language="it-IT"):
        self.language=language
        self.verbose=verbose

        self.PROMPT_LIMIT = 5
        
        #return code
        self.ERROR=0
        self.SUCCESS=1
        self.UNDEFINED=2
        self.QUIT=3

        #stop condition
        self.stop_condition="esci"
        
        # create recognizer and mic instances
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def start(self):
        for j in range(self.PROMPT_LIMIT):
            guess = recognize_speech_from_mic(self.recognizer, self.microphone,self.language)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            if j==self.PROMPT_LIMIT-1:
                return self.UNDEFINED,""


        # if there was an error, stop 
        if guess["error"]:
            if self.verbose:
                print("ERROR: {}".format(guess["error"]))
            return self.ERROR,""
        
        if self.verbose:
            print("Hai detto: {}".format(guess["transcription"]))
        text=guess["transcription"].lower()
        flag=self.SUCCESS
        if text==self.stop_condition:
            flag=self.QUIT
        return flag,text

def main():
    sr = Speech_to_text()
    text=""

    while text!=sr.stop_condition:
        flag,text=sr.start()
        print(text)
