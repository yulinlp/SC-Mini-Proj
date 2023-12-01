import zhconv
import re
import numpy as np
import pandas as pd
import jieba_fast
from tqdm import tqdm
import multiprocessing 
from gensim.models import KeyedVectors
from config import Config
from transformers import AutoTokenizer

class preprocessing():
    
    def __init__(self, data_path, param):
        self.stop_words = self.get_stop_words()  
        self.data_path = data_path
        self.word2vec = KeyedVectors.load_word2vec_format(Config.Word2Vec_model_path)
        self.param = param
        
    # 数据预处理
    def helper(self, data):
        max_len = self.param['sentence_max_length']
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        data = zhconv.convert(data, 'zh-cn')
        data = re.sub(pattern,'', data)
        data = jieba_fast.lcut(data)
        # print('data初步处理完成',len(data),type(data[0]))
        # data是一段话的词语列表
        tokenized_data = []
        default_value = np.zeros(self.param['embed_dim'])
        
        for word in data:
            if word not in self.stop_words:
                try :
                    tokenized_data.append(self.word2vec[word].astype(np.float32)) # dim = 1,len = 200
                except Exception:
                    tokenized_data.append(default_value)
    #     data = [word2vec[word] for word in data if word not in stop_words]
        # print("初步向量化结束",len(tokenized_data),type(tokenized_data[0]))
        
        if len(tokenized_data) >= max_len:
            tokenized_data = np.array(tokenized_data[:max_len],dtype=np.float32)
        else:
            tokenized_data = np.pad(tokenized_data, ((0, max_len - len(tokenized_data)), (0, 0)), 'constant', constant_values=0)
        # print("data处理完成",tokenized_data.shape,type(tokenized_data))
        return tokenized_data
    
            
    def run(self):
        raw_data = pd.read_csv(self.data_path)
        raw_x = raw_data.iloc[:, 1].tolist()  #list(105000)
        # pool = multiprocessing.Pool(processes=2)
        # x = pool.map(self.helper, tqdm(raw_x))
        # # print(len(x),type(x))
        # pool.close()
        # pool.join()
        x = [self.helper(item) for item in tqdm(raw_x, desc='Processing Data_X')]
        
        if self.data_path == Config.test_path:
            return x
        
        raw_y = raw_data.iloc[:, 2:].T
        y = []
        for col in tqdm(raw_y.columns, desc='Processing Data_Y'):
            y.append([i+2 for i in raw_y[col].values.tolist()])
        return x, y

    def get_stop_words(self):
        stop_words = set()
        with open(Config.stop_word_path,'r',encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
        
        return stop_words


def Bert_Preprocessing(data_path, param):
    raw_data = pd.read_csv(data_path)
    raw_x = raw_data.iloc[:, 1].tolist()  #list(105000)
    
    tokenizer = AutoTokenizer.from_pretrained(param['tokenizer_path'])
    ids_x = tokenizer(raw_x, truncation=True, max_length=param['sentence_max_length'],
                            padding='max_length', return_tensors='pt')
    x = ids_x['input_ids']
    
    raw_y = raw_data.iloc[:, 2:].T
    y = []
    for col in tqdm(raw_y.columns, desc='Processing Data_Y'):
        y.append([i+2 for i in raw_y[col].values.tolist()])
    return x, y