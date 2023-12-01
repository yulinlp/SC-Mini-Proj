import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import gc
import warnings
from models.My_TextCNN import TextCNN
from models.My_LSTM import LSTM
from models.My_BERT import BERT
from config import Config
from utils.strategy import fetch_scheduler, set_seed
from utils.data_loader import get_test_DataLoader, get_train_valid_Dataloader
from train import train_one_epoch
from valid import valid_one_epoch
from test import test
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./log/demo")
model_name = Config.model
model_train = Config.train[model_name]
model_param = Config.param[model_name]
Bert = BERT()
Bert.load_state_dict(torch.load('/home/CodeSpace/Proj1/results/Best_BERT.pth'))
models = {
    'LSTM': LSTM(Bi = True),
    'TextCNN': TextCNN(),
    'BERT': Bert
}

log = open(Config.logging_path, "a")
sys.stdout = log
print("\n\n\n\n=======================================================================================")
currentDateAndTime = datetime.now()
print("The current date and time is", currentDateAndTime)
print('\n')
print(model_param)
print(model_train)
print('\n')

gc.collect()                        # 清空内存垃圾
torch.cuda.empty_cache()            # 清空显存
warnings.filterwarnings('ignore')   # 别警告我
set_seed(2023)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models[model_name]
# model.init_weight()

optimizer = AdamW(model.parameters(), lr=model_train['lr'], weight_decay=model_train['weight_decay'])
criterion = nn.CrossEntropyLoss()
scheduler = fetch_scheduler(optimizer = optimizer, schedule = model_train['schedule'])
model_save_path = Config.model_save_path

# 获取DataLoader类型的训练数据
train_dl, valid_dl = get_train_valid_Dataloader(model_train['batch_size'],param=model_param, isBert=model_name == 'BERT')

# 训练n个epoch，保存最佳模型权重，查看最佳：1.验证损失 2.验证准确率
model.to(device)
best_model_state = copy.deepcopy(model.state_dict())
best_valid_loss = np.inf
best_valid_accuracy = 0.0
best_valid_F1 = 0.0
start_time = time.time()

for epoch in range(1, model_train['epoch']+1):
    # 进行一轮训练与验证
    train_loss, train_accuracy, train_F1 = train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch, model_param, writer)
    valid_loss, valid_accuracy, valid_F1 = valid_one_epoch(model, criterion, valid_dl, device, epoch, model_param, writer)
    scheduler.step()
    # 如果验证损失降低了，则：1.保存模型状态 2.更新最佳验证损失 3.更新最佳验证准确率
    if valid_F1 > best_valid_F1:
        print(f'best valid F1 has improved ({best_valid_F1}---->{valid_F1})')
        best_valid_F1 = valid_F1
        best_valid_loss = valid_loss
        best_valid_accuracy = valid_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        if model_name == 'BERT':
            model.save()
        torch.save(best_model_state, model_save_path)
        print('A new best model state  has saved')
        
writer.close()        
end_time = time.time()
print('Training Finish !!!!!!!!')
print(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}, best valid F1 == {best_valid_F1}')
time_cost = end_time - start_time
print(f'training cost time == {time_cost}s')