# 创建数据集
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from config import Config
from utils.preprocessing import preprocessing, Bert_Preprocessing

class MyDataset(Dataset):
    def __init__(self, x, y=None):
        self.x= x
        self.test = 0
        if y is not None:
            self.y = np.array(y)
            self.test = 1
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if not self.test:
            return {'x': self.x[idx]}
        return {'x': self.x[idx], 'y': self.y[idx]}
    
        
def get_train_valid_Dataloader(batch_size, param, isBert = False):
    if isBert:
        train_data = Bert_Preprocessing(Config.train_path, param=param)
    else:
        train_data = preprocessing(Config.train_path, param=param).run()
    train_set = MyDataset(train_data[0], train_data[1])
    # print("trainset第一个元素是: ",train_set[0])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
    print("训练数据加载完成")
    
    if isBert:
        valid_data = Bert_Preprocessing(Config.valid_path, param=param)
    else:
        valid_data = preprocessing(Config.valid_path, param=param).run()
    valid_set = MyDataset(valid_data[0], valid_data[1])
    valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True, pin_memory = True)
    print("验证数据加载完成")
    
    return train_loader, valid_loader


def get_test_DataLoader(batch_size, param):
    test_set = MyDataset(preprocessing(Config.test_path, param=param).run())
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, pin_memory = True)
    print('测试数据加载完成')
    
    return test_loader       