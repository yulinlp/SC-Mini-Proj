import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import Config

class TextCNN(nn.Module):
    def __init__(self, param = Config.param['TextCNN']):
        super(TextCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        
        self.class_num = param['class_num']
        self.label_num = param['label_num']
        self.param = param
        
        self.conv1 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv2 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv3 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, self.label_num * self.class_num)
    
    def conv_and_pool(self, x, conv):
        # x: (batch_size, 1, sentence_length, embed_dim)
        x = conv(x.float())
        # x: (batch_size, kernel_num, sentence_length - kernel_size + 1, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch_size, kernel_num, sentence_length - kernel_size + 1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch_size, kernel_num)
        return x
    
    def forward(self, x):
        # x: (batch_size, sentence_length, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch_size, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)
        # xi: (batch_size, kernel_num)
        x = torch.cat((x1,x2,x3), 1)
        # x: (batch_size, kernel_num*3)
        x = self.dropout(x)
        x = self.fc(x)
        # x: (batch_size, num_labels)
        x = x.view(-1, self.class_num, self.label_num)
        # x: (batch_size, num_labels, num_classes)
        
        # logits = torch.softmax(x,dim = 1)
        # logits: (batch_size, num_labels)
        return x
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()