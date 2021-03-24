import torch
import numpy as np # linear algebra
import torch.nn as nn
import os
#!pip3 install transformers==3
import transformers
from transformers import BertModel, BertTokenizer
import warnings

# ignore warning
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# define bert
berts={"it-IT":"bert-base-multilingual-cased", "en-US":"bert-base-uncased"}  

# define maximum sequence length
max_seq_len=9


try:
    from google.colab import drive
    drive.mount("/content/drive/")
    data_dir_intent_classification = "/content/drive/My Drive/Tesi/Code/Text_production/"
except:
    data_dir_intent_classification = "./"
    if __package__:
        data_dir_intent_classification +=__package__

class BERT_Arch(nn.Module):
    def __init__(self, bert,label_map):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,len(label_map))

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)
        return x

class Intent_classification:
    def __init__(self,verbose=False,device_type="cpu",language="it-IT"):
        path = os.path.join(data_dir_intent_classification,"Weights","topic_saved_weights.pt")
        self.verbose=verbose
        self.device = torch.device(device_type)
        print(device_type)
        checkpoint = torch.load(path,map_location=self.device)
        self.predictor = checkpoint.get("model")
        self.tokenizer = BertTokenizer.from_pretrained(berts[language])
        self.tag = checkpoint.get("id_map")

    def predict(self,text):
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids + [0] * (max_seq_len-len(input_ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)

        input_mask = [1]*len(tokens) + [0] * (max_seq_len - len(tokens))
        input_mask = torch.tensor(input_mask).unsqueeze(0)
        input_mask = input_mask.to(self.device)

        logits = self.predictor(input_ids,input_mask)
        prob = torch.nn.functional.softmax(logits,dim=1)
        result = [(self.tag[idx],item *100) for idx,item in enumerate(prob[0].tolist())]
        if self.verbose:
            print("Intents: {}".format([(i[0],round(i[1],4)) for i in result]))
        #preds = logits.detach().cpu().numpy()
        #pred_val = np.argmax(preds)
        #pred_val = self.tag[pred_val]
        best_intent = max(result, key=lambda x: x[1])
        return result,best_intent  




def main():
  pred = Intent_classification()

  list_input = [
      'di che colore è',
      'che tessitura ha questo oggetto',
      'che cosa è questo',
      'che forma ha',
      'cosa è',
      'è un toroide'] 

  for item in list_input:
      confidences,pred_val = pred.predict(item)
      prob = round([i for i in confidences if i[0]==pred_val][0][1],4)
      print("'" + item + "' = " + pred_val + ": " + str(prob))
      print([(i[0],round(i[1],4)) for i in confidences])
      print()

if __name__ == "__main__":
    main()  
