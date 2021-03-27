import cv2
import random

class Shape_extractor:
    def extract(self,image):
        shapes=None
        return self.classify(shapes),shapes

    def classify(self,features):
        labels=["tondo", "quadrato", "esagono", "triangolo", "rettangolo", "ovale"]
        return [(l,random.random()) for l in labels]    
