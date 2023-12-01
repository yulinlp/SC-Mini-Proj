import torch
import torch.nn as nn
from config import Config

# 参考地址: https://zhuanlan.zhihu.com/p/606019832
class LSTMCell(nn.Module):
    def __init__(self, param = Config.param['LSTM'], is_first = False):
        super().__init__()
        
        hidden_size = param['hidden_size']
        input_size = param['hidden_size']
        if is_first:
            input_size = param['embed_dim']

        self.hidden = nn.Linear(hidden_size, 4*hidden_size)
        self.activate = nn.Linear(hidden_size*8, hidden_size*4)
        self.input = nn.Linear(input_size, 4*hidden_size)
        self.dropout = nn.Dropout(param['dropout'])

    def forward(self, x, h, c):        
        gates = torch.concat((self.hidden(h), self.input(x)),dim=-1)
        i, f, g, o = self.activate(gates).chunk(4, dim=-1)
        
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        c_next = self.dropout(c_next)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        h_next = self.dropout(h_next)
        
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, param = Config.param['LSTM'], Bi = False):
        super().__init__()
        self.Bi = Bi
        self.class_num = param['class_num']
        self.label_num = param['label_num']
        self.sentence_length = param['sentence_max_length']
        self.hidden_size = param['hidden_size']
        self.n_layers = param['n_layers']
        self.dropout = nn.Dropout(param['dropout'])
        self.cells = nn.ModuleList([LSTMCell(is_first=True)] + [LSTMCell() for _ in range(self.n_layers - 1)])
        self.bi_cells = nn.ModuleList([LSTMCell(is_first=True)] + [LSTMCell() for _ in range(self.n_layers - 1)])
        self.hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * self.sentence_length, self.label_num * self.class_num)

        
    def forward(self, x, state=None, Bi_state=None):
        # x: (batch_size, sentence_length, embed_dim)
        # state 包含了传递过来的state即(h, c)
        # h和c各自的形状为[batch_size, hidden_size]
        x = x.to(torch.float32)
        batch_size = x.shape[0]
        
        if state is None:
            h = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(self.n_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(self.n_layers)]
        else:
            h,c = state
        
        if self.Bi:
            if Bi_state is None:
                Bi_h = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(self.n_layers)]
                Bi_c = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(self.n_layers)]
            else:
                Bi_h,Bi_c = state
                
        out = []
        for word_idx in range(self.sentence_length):
            input = x[:,word_idx,:]
            Bi_input = x[:,self.sentence_length - word_idx - 1,:]
            for layer_idx in range(self.n_layers):    
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](input, h[layer_idx], c[layer_idx])
                input = self.dropout(h[layer_idx])
                if self.Bi:
                    Bi_h[layer_idx], Bi_c[layer_idx] = self.bi_cells[layer_idx](Bi_input, Bi_h[layer_idx], Bi_c[layer_idx])
                    Bi_input = self.dropout(Bi_h[layer_idx])
            if self.Bi:
                out.append(torch.relu(self.hidden(h[-1])) + torch.relu(self.hidden(Bi_h[-1])))
            else:
                out.append(h[-1])
        
        out = torch.stack(out).transpose_(0,1)
        out = out.reshape(out.shape[0], -1)    # [batch_size, sentence_length, hidden_size]
        # h = torch.stack(h)
        # c = torch.stack(c)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(-1, self.class_num, self.label_num)
        
        return out
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal(m.weight)