from Grounding.Extractors.Color_extractor import Color_extractor
from Grounding.Extractors.Shape_extractor import Shape_extractor
from Grounding.Extractors.Texture_extractor import Texture_extractor
import random

class Grounding:
    def __init__(self,verbose=False):
        self.verbose=verbose
        self.color_extractor=Color_extractor()
        self.shape_extractor=Shape_extractor()
        self.texture_extractor=Texture_extractor()
    def classify(self, scan, intent):
        image,depth=scan
        features={}
        features['color']=self.color_extractor.extract(image)
        features['shape']=self.shape_extractor.extract(image)
        features['texture']=self.texture_extractor.extract(image)
        features['general']=sorted([(i, random.random()) for i in ["palla da tennis", "uovo", "banana", "pera", "anguria", "anguilla"]],key=lambda x:x[1])
         
        if self.verbose:
            print("Intent: {}".format(intent))
            print("Color: {}\nColor features: {}\n".format(features['color'][0],features['color'][1])) 
            print("Shape: {}\nShape features: {}\n".format(features['shape'][0],features['shape'][1])) 
            print("Texture: {}\nTexture features: {}\n".format(features['texture'][0],features['texture'][1]))         
        return features[intent[:-6]][0]


    def learn(self, scan, intent, label):
        image,depth=scan
        return    
