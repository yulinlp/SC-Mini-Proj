import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import Config

@torch.no_grad()
def test(model, criterion, test_loader, device, model_path):
    model.eval()
    num_labels = Config.TextCNN_param['label_num']
    num_examples = 0    # 样本数
    num_F1 = 0
    total_loss = 0
    total_correct = 0
    total_F1 = 0
    
    # 加载模型权重开始测试
    model.load_state(torch.load(model_path))
    bar = tqdm(enumerate(test_loader), total = len(test_loader))
    for i, batch in bar:
        x = batch['x'].to(device=device)
        y = batch['y'].to(device=device)
        # 获取模型输出并计算损失
        out = model(x) # (batch_size:128, num_classes:4, num_labels:20)
        loss = criterion(out, y.long())
        
        # 计算当前batch中：1.预测正确的样本总数 2.总损失
        num_examples += len(y) * num_labels
        preds = out.argmax(dim=1)
        
        correct = (preds == y).sum().item()
        total_correct += correct
        total_loss += loss.item()
        
        y = y.cpu().numpy().T
        preds = preds.cpu().numpy().T
        F1_score = 0
        
        for each_y, each_preds in zip(y,preds):
            f1 = f1_score(each_y, each_preds, average='macro')
            F1_score += f1 / num_labels
        total_F1 += F1_score
        num_F1 += 1
        
        # 计算准确率和平均损失，并显示在tqdm中
        accuracy = total_correct / num_examples
        avg_loss = total_loss / num_examples
        bar.set_postfix(current_loss=avg_loss, current_accuracy=accuracy, F1_score=F1_score)
    
    # 返回整个测试集的平均损失和测试准确率和MacF1
    return total_loss/num_examples, total_correct/num_examples, total_F1/num_F1