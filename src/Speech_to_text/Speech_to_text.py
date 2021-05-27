import random
import time
import os
import sys
import speech_recognition as sr
from contextlib import contextmanager

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stderr_redirected(to=os.devnull, stderr=None):
    if stderr is None:
       stderr = sys.stderr

    stderr_fd = fileno(stderr)
    # copy stderr_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied: 
        stderr.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stderr_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stderr_fd)  # $ exec > to
        try:
            yield stderr # allow code to be run with the redirected stderr
        finally:
            # restore stderr to its previous value
            #NOTE: dup2 makes stderr_fd inheritable unconditionally
            stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)  # $ exec >&copied


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
        
        # create recognizer and mic instances
        sr.SAMPLE_RATE = 48000
        with stderr_redirected(to=os.devnull):
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
            print("Hai detto: {}".format(guess["transcription"].lower()))
        text=guess["transcription"].lower()
        flag=self.SUCCESS
        return flag,text

def main():
    s = Speech_to_text(verbose=True)
    text=""
    flag=s.ERROR    
    while flag!=s.SUCCESS or text!="esci":
        print("Sto ascoltando...")
        flag,text=s.start()
        if flag==s.ERROR:
            print("error")
            return 
        elif flag==s.UNDEFINED:
            print("undefined")

if __name__=="__main__":
    main()        
