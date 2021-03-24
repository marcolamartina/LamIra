import cv2
import random

class Texture_extractor:
    def extract(self,image):
        textures=None
        return self.classify(textures),textures

    def classify(self,features):
        labels=["ruvido", "liscio", "intrecciato", "nido d'ape", "damascato", "feltro"]
        return [(l,random.random()) for l in labels]    
