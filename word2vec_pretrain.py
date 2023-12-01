# 用word2vec训练词向量
import zhconv
import jieba_fast
import pandas as pd
import re
import multiprocessing
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from config import Config

def helper(data):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    data = zhconv.convert(data, 'zh-cn')
    data = re.sub(pattern,'', data)
    return jieba_fast.lcut(data)

class word2vec(): 
    def __init__(self):
        extra_data = pd.read_csv(Config.pretrain_data_path,encoding='utf-8').dropna(axis=0, how='any')
        extra_data = extra_data['comment'].tolist()
        test_data = pd.read_csv(Config.test_path, encoding='utf-8').dropna(axis=0, how='any')
        test_data = test_data.iloc[:, 1].tolist()
        train_data = pd.read_csv(Config.train_path, encoding='utf-8').dropna(axis=0, how='any')
        train_data = train_data.iloc[:, 1].tolist()
        valid_data = pd.read_csv(Config.valid_path, encoding='utf-8').dropna(axis=0, how='any')
        valid_data = valid_data.iloc[:, 1].tolist()
        print(type(test_data),type(extra_data))
        corpus = extra_data + test_data + train_data + valid_data
        self.corpus = corpus        
        # 超参数
        self.num_features = 200
        self.context = 5
        
        print("完成初始化")
    
    def clean_text(self):        
        pool = multiprocessing.Pool(processes=4) 
    
        x = pool.map(helper, tqdm(self.corpus))

        pool.close()
        pool.join()
        
        print("完成clean_text")
        
        return x
    
    def get_model(self):
        """
        从头训练word2vec模型
        :param text: 经过清洗之后的语料数据
        :return: word2vec模型
        重要参数含义
        :param num_features:  返回的向量长度
        :param min_word_count:  最低词频
        :param context: 滑动窗口大小
        """
        text = self.clean_text()
        print("开始训练")
        model = Word2Vec(text, vector_size=self.num_features, window=self.context, workers=multiprocessing.cpu_count() * 2)
        print("训练完成")
        return model
    
word2vec_ = word2vec()
model = word2vec_.get_model()
model.wv.save_word2vec_format(Config.Word2Vec_save_path, binary=False)

# 获取词的词向量
print('测试开始')
a = model.wv['我']
print(a)