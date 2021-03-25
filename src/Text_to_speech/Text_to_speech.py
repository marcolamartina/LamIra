from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from ctypes import *
from contextlib import contextmanager
import pyaudio
from datetime import datetime
import platform


mac=platform.system()=="Darwin"
if not mac:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    if mac:
        yield
        return
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

class Text_to_speech:
    def __init__(self,verbose=False,language="it-IT"):
        self.language=language
        self.verbose=verbose

    def speak(self,text):    
        with noalsaerr() as n:
            # Define a BytesIO object
            mp3_fp = BytesIO()

            # Passing the text and language to the engine,  
            # here we have marked slow=False. Which tells  
            # the module that the converted audio should  
            # have a high speed 
            tts = gTTS(text=text, lang=self.language[:2],slow=False)

            # Write audio in BytesIO object
            tts.write_to_fp(mp3_fp)

            # Setting the seek
            mp3_fp.seek(0)

            # Decode the audio format
            song = AudioSegment.from_file(mp3_fp, format="mp3")

            #Transcript
            if self.verbose:
                print(text)

            # Play audio
            play(song)

    def save(self,text, filename):
        with noalsaerr() as n:
            # Passing the text and language to the engine,  
            # here we have marked slow=False. Which tells  
            # the module that the converted audio should  
            # have a high speed         
            tts = gTTS(text=text, lang=self.language[:2],slow=False)

            tts.save(filename+".mp3")
        
    def speak_from_file(self, filename):
        with noalsaerr() as n:
            # Decode the audio format
            song = AudioSegment.from_file("../Media/Audio/"+filename, format="mp3")

            # Play audio
            play(song)


def save_audio():
    text_to_speech=Text_to_speech(verbose=False,language="it-IT")
    nome="LamIra"
    paths=[r'/Users/marco/Google Drive/Tesi/Media/Audio/',r'/Users/marco/GitHub/LamIra/Media/Audio/']
    welcome_message_list=["Ciao, "+nome+" ti dà il benvenuto.", "Benvenuto in "+nome+".", nome+" ti dà il benvenuto", "Ciao, sono "+nome+" e sono qui per aiutarti", "Sono "+nome+" e voglio darti il benvenuto."]
    quit_message_list=["Arrivederci da "+nome, "È stato un piacere", nome+" ti ringrazia", "Grazie per aver scelto "+nome+", arrivederci.", nome+" ti augura una buona giornata!", "A presto", "Un saluto da "+nome]
    error_message_list=["C'è stato un errore imprevisto. Devo spegnermi.", "Si è verificato un errore. Per favore riprova", "Ho riscontrato un errore.", "Per favore riprova", "Scusami, ho una perdita di bit, riprova", "Ho subito un attacco hacker e devo riavviarmi", "Ci stanno tracciando, staccah staccah tutte cose!"]
    undefined_message_list=["Non ho ben capito. Che hai detto?", "Mi spiace ho dei dubbi, puoi ripetere?", "Puoi ripetere quello che hai detto?", "Scusami, non ho compreso quello che hai detto, puoi ripetere?", "I miei microfoni sono sporchi, devo andare dall'hexorino, puoi dirlo ad alta voce?", "Non ti ho sentito, mi sa che devo comprarmi un amplifon. Puoi ripetere?"]
    query_message_list=["Come posso aiutarti?", "Come posso esserti utile?", "In cosa posso aiutarti?", "Di cosa hai bisogno?", "Cosa posso fare per te?"]
    cannot_answer_message_list=["Purtroppo non posso rispondere alla tua domanda.", "Ancora non so aiutarti.", "Non posso esaudire la tua richiesta.", "Non so come aiutarti", "Non riesco a trovare una risposta alla tua domanda.", "Scusami, non posso aiutarti."]
    training_mode_list=["Hai attivato la modalità addestramento", "Sei entrato nella modalità addestramento", "Benvenuto nella modalità addestramento" ]
    query_mode_list=[s.replace("addestramento","query") for s in training_mode_list]
    training_request_list=["Cosa vuoi insegnarmi?","Cosa posso imparare?","Cosa vuoi farmi vedere?","Cosa posso apprendere?"]
    
    dict_message={"Welcome":welcome_message_list,"Quit":quit_message_list,"Error":error_message_list,"Undefined":undefined_message_list, "Query":query_message_list, "Cannot_answer":cannot_answer_message_list, "Training_mode":training_mode_list, "Query_mode":query_mode_list, "Training_request":training_request_list}
    for folder,message_list in dict_message.items():
        for i,message in enumerate(message_list):
            for path in paths:
                text_to_speech.save(message, path+folder+"/"+folder.lower()+"_message_"+str(i))


