## SC情感分析 专业实践 Task1
### 1. 项目结构：
.  
├── config.py  
├── dataset  
│   ├── other_resources  
│   │   ├── hit_stopwords.txt  
│   │   ├── ratings.csv.zip  
│   │   └── word2vec  
│   │       ├── word2vec_first.model  
│   │       ├── word2vec_L.model  
│   │       ├── word2vec_M.model  
│   │       └── word2vec_S.model  
│   ├── test_data  
│   │   ├── protocol.txt  
│   │   ├── README.txt  
│   │   └── sentiment_analysis_testa.csv  
│   ├── train_data  
│   │   ├── ~$ntiment_analysis_trainingset_annotations.docx  
│   │   ├── protocol.txt  
│   │   ├── README.txt  
│   │   ├── sentiment_analysis_trainingset_annotations.docx  
│   │   └── sentiment_analysis_trainingset.csv  
│   └── val_data  
│       ├── protocol.txt  
│       ├── README.txt  
│       ├── sentiment_analysis_validationset_annotations.docx  
│       └── sentiment_analysis_validationset.csv  
├── log  
├── models  
│   ├── My_BERT.py  
│   ├── My_LSTM.py  
│   ├── My_TextCNN.py  
│   └── __pycache__  
│       ├── My_BERT.cpython-38.pyc  
│       ├── My_LSTM.cpython-38.pyc  
│       └── My_TextCNN.cpython-38.pyc  
├── __pycache__  
│   ├── config.cpython-38.pyc  
│   ├── test.cpython-38.pyc  
│   ├── train.cpython-38.pyc  
│   └── valid.cpython-38.pyc  
├── README.md  
├── results  
│   ├── BERT.log  
│   ├── Best_BERT.pth  
│   ├── Best_LSTM.pth  
│   ├── Best_TextCNN.pth  
│   ├── LSTM.log  
│   └── TextCNN.log  
├── run.py  
├── test.py  
├── train.py  
├── tree.txt  
├── utils  
│   ├── data_loader.py  
│   ├── preprocessing.py  
│   ├── __pycache__  
│   │   ├── data_loader.cpython-38.pyc  
│   │   ├── preprocessing.cpython-38.pyc  
│   │   └── strategy.cpython-38.pyc  
│   └── strategy.py  
├── valid.py  
└── word2vec_pretrain.py  

13 directories, 48 files  

### 2. 项目说明
见 `专业实践：情感分类技术实践.pdf`

### 3. 数据集说明
- `dataset/test_data` 、 `dataset/val_data` 、 `dataset/train_data` 来自于 `AIchallenger2018` 竞赛， 详见 **2.项目说明** 。
- `other_resources/ratings.csv.zip` 来自于 `https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_dianping/intro.ipynb`, 为大众点评 `canguan` 数据集。该数据集作为 `word2vec` 的部分训练集。
- `other_resources/hit_stopwords.txt` 来自于 `https://github.com/goto456/stopwords/blob/master/hit_stopwords.txt`, 为中文停用词表, 用于清洗数据。

### 4. 模型文件说明
- `other_resources/word2vec/` 包含三个模型:
  1. `word2vec_L`: 516.71MB, 由`canguan`完整数据集（共4422473条数据） + 竞赛完整数据集训练得到
  2. `word2vec_M`: 361.91MB, 由`canguan`数据集前2000000条数据 + 竞赛完整数据集训练得到
  3. `word2vec_S`: 145.87MB, 只由竞赛完整数据集训练得到
  4. `word2vec_first`: 392.00MB, 由canguan数据集前3000000条数据训练得到。 若无特殊说明，TextCNN以及LSTM的绝大多数实验都是用它进行词向量嵌入的
- `results/` 包含三个模型, 均为对应实验最终调得的最优模型

### 5. 超参数说明
详见 `config.py`

### 6. 实验结果说明
- `results/` 保存了训练过程中的log文件  
后续会更新实验的全过程




