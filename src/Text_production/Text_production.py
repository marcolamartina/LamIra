import os
import random
import re

local=False
verbose=False    

try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_text_production = "/content/drive/My Drive/Tesi/Code/Text_production/"
except:
    data_dir_text_production = os.path.dirname(__file__)
    

class Text_production:
    def __init__(self,verbose=False):
        self.__set_labels()
        self.verbose=verbose
        self.local=local
        self.spaces=["l'oggetto","il colore","la forma","la tessitura"]        

    def set_attributes_predictions(self,intent_label,predictions):
        self.intent_label=intent_label
        self.predictions=Predictions(predictions)

    def concatenate_labels(self,labels):
        if len(labels)<=1:
            return labels[0]
        return " e ".join([", ".join(labels[:-1]),labels[-1]])

    def split_label(self,label):
        elements=label.split("-")
        return self.concatenate_labels(elements)
        
    def to_text_predictions(self,intent_label,predictions,description):
        self.set_attributes_predictions(intent_label,predictions)
        path = os.path.join(data_dir_text_production,"Templates",self.intent_label,"{}.txt".format(self.predictions.prediction_type))
        with open(path,"r") as f:
            outputs=[]
            for line in f.readlines():
                outputs.append(line.strip())
        output=random.choice(outputs)
        if not self.predictions.prediction_type=="cannot_answer":
            output=output.replace("{label}",self.split_label(self.predictions.best_prediction.label.label))
            output=output.replace("{art_def}",self.predictions.best_prediction.label.article_definite) 
            output=output.replace("{art_indef}",self.predictions.best_prediction.label.article_indefinite)
            if self.predictions.prediction_type=="dubious_answer":
                output=output.replace("{label_a}",self.split_label(self.predictions.best_predictions[1].label.label)) 
                output=output.replace("{art_def_a}",self.predictions.best_predictions[1].label.article_definite)
                output=output.replace("{art_indef_a}",self.predictions.best_predictions[1].label.article_indefinite)
        if description:
            if self.predictions.prediction_type=="cannot_answer":
                output+=", però posso dirti che è di colore {}, è a forma di {} ed è di un materiale {}".format(*description)
            if self.predictions.prediction_type in ["dubious_answer","unsure_answer"]:
                output+=", poiché è di colore {}, è a forma di {} ed è di un materiale {}".format(*description)   
        if self.verbose:
            print("Best predictions: {}".format([(p.label.label,round(p.confidence,4)) for p in self.predictions.best_predictions]))
        return output

    def to_text_subject(self,subjects):
        subjects_list=[" ".join(self.spaces[i],s) for i,s in enumerate(subjects) if s ]
        outputs=["Ho appena imparato","Ho imparato","Ho appreso"]
        output=random.choice(outputs) 
        output+=" "+self.concatenate_labels(subjects_list)
        return output             

    def __set_labels(self):
        path = os.path.join(data_dir_text_production,"..","Intent_Classification","Dataset","Train")
        try:
            files = os.listdir( path )
            self.labels=[f[:-4] for f in files]
        except FileNotFoundError:
            self.labels=['texture_query', 'shape_query', 'general_query', 'color_query']       

class Predictions():
    def __init__(self,predictions):
        self.similarity_threshold=0.3
        self.labels,self.confidences=zip(*predictions)
        self.__set_predictions(predictions)        

    def __set_predictions(self,predictions):
        #couples answer_type-threshold
        prediction_types=[("sure_answer",0.6),("unsure_answer",0.4),("dubious_answer",0.3),("cannot_answer",0)] 

        #list of possible predictions sorted by confidence
        self.predictions=sorted([Prediction(l,c) for (l,c) in predictions],key=lambda x: x.confidence,reverse=True) #
        self.best_prediction=self.predictions[0]

        #setting prediction type
        for prediction_type in prediction_types:
            if self.best_prediction.confidence>prediction_type[1]:
                self.prediction_type=prediction_type[0]
                break

        #list of best predictions
        self.best_predictions = [p for p in self.predictions if self.best_prediction.confidence-p.confidence<self.similarity_threshold]  
        #change prediction type if there aren't similar labels
        if self.prediction_type=="dubious_answer" and len(self.best_predictions)<2:
            self.prediction_type="cannot_answer"         

class Prediction():
    def __init__(self,label,confidence):
        self.label=Label(label)
        self.confidence=confidence
        self.prediction=(self.label,self.confidence)

    def __str__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, Prediction):
            return self.label == other.label
        return False                

class Label():
    def __init__(self,label):
        self.label=label
        self.gender=self.__get_gender()
        self.singular=self.__is_singular()
        self.article_definite=self.__get_article_definite()
        self.article_indefinite=self.__get_article_indefinite()

    def __is_singular(self):
        if self.label.split()[0][-1]=="ie":
            return False
        else:
            return True    

    def __get_gender(self):
        if self.label[-1].split()[0] in "ae":
            return "F"
        else:
            return "M"

    def __get_article_definite(self):
        if self.singular:
            if self.label[0].lower() in "aeiou":
                return "l'"
            if self.gender=="F":
                return "la "
            if re.search("^ps.+|^pn.+|^gn.+|^z.+|^x.+|^y.+|^s[^aeiou].+|^i[aeiou].+",self.label):  
                return "lo "
            else:
                return "il "
        else:
            if self.gender=="F":
                return "le "
            if re.search("^ps.+|^pn.+|^gn.+|^z.+|^x.+|^y.+|^s[^aeiou].+|^i[aeiou].+",self.label):  
                return "gli "
            else:
                return "i "

    def __get_article_indefinite(self):
        if self.singular:
            if self.label[0].lower() in "aeiou" and self.gender=="F":
                return "un'"
            if self.gender=="F":
                return "una "
            if re.search("^ps.+|^pn.+|^gn.+|^z.+|^x.+|^y.+|^s[^aeiou].+|^i[aeiou].+",self.label):  
                return "uno "
            else:
                return "un "
        else:
            if self.gender=="F":
                return "delle "
            if re.search("^ps.+|^pn.+|^gn.+|^z.+|^x.+|^y.+|^s[^aeiou].+|^i[aeiou].+",self.label):  
                return "degli "
            else:
                return "dei "

    def with_article_definite(self):
        return self.article_definite+self.label

    def with_article_indefinite(self):
        return self.article_indefinite+self.label

    def __str__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, Label):
            return self.label == other.label
        return False

def main():
  predictions=[
      ("general_query",[("palla da tennis",0.83),("uovo",0.86),("arancia",0.81)]),
      ("shape_query",[("rettangolo",0.9),("cerchio",0.1),("trapezio",0.51)]),
      ("color_query",[("rosso-bianco",0.24),("viola-giallo-fucsia",0.86),("giallo",0.15)]),
      ("color_query",[("blu-bianco",0.51),("viola-nero-fucsia",0.48),("giallo",0.01)]),
      ("texture_query",[("liscio",0.03),("ruvido",0.26),("setoso",0.31)])
      ]

  for prediction in predictions:
      tp=Text_production()
      print(tp.to_text_predictions(*prediction))

if __name__ == "__main__":
    main()  
