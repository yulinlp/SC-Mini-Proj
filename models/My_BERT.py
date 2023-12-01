from transformers import AutoModel
import torch.nn as nn
import torch
from config import Config

class BERT(nn.Module):
    def __init__(self, param=Config.param['BERT'], finetuned = True) -> None:
        super().__init__()
        bert_path = param['model_path']
        if finetuned:
            bert_path = Config.BERT_save_path
        self.bert = AutoModel.from_pretrained(bert_path)
        self.label_num = param['label_num']
        self.class_num = param['class_num']
        self.fc = nn.Linear(768, self.label_num * self.class_num)
        self.dropout = nn.Dropout(param['dropout'])
    
    def forward(self, x):
        # with torch.no_grad():
        x = self.bert(x).pooler_output
        x = self.dropout(x)
        x = self.fc(x)
        
        out = x.view(-1, self.class_num, self.label_num)
        
        return out
    
    def save(self):
        self.bert.save_pretrained(Config.BERT_save_path)