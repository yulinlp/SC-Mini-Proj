class Config():
    
    model = 'BERT'
    
    root_path = '/home/CodeSpace/Proj1/'
    test_path = 'dataset/test_data/sentiment_analysis_testa.csv'
    train_path = 'dataset/train_data/sentiment_analysis_trainingset.csv'
    valid_path = 'dataset/val_data/sentiment_analysis_validationset.csv'
    pretrain_data_path = 'dataset/other_resources/canguan_1.8GB.csv'    
    stop_word_path = 'dataset/other_resources/hit_stopwords.txt'
    logging_path = 'results/' + model + '.log'
    Word2Vec_model_path = 'dataset/other_resources/word2vec/word2vec_S.model'
    Word2Vec_save_path = 'dataset/other_resources/word2vec/new_word2vec.model'
    model_save_path = 'results/' + model + '.pth'
    BERT_save_path = 'dataset/other_resources/finetuned_BERT'
    
    test_path = root_path + test_path         # 测试数据存放路径
    train_path = root_path + train_path       # 训练数据存放路径
    valid_path = root_path + valid_path       # 验证数据存放路径
    pretrain_data_path = root_path + pretrain_data_path  # 训练tokenizer的语料存放路径
    stop_word_path = root_path + stop_word_path  #停用词表存放路径
    logging_path = root_path + logging_path
    Word2Vec_model_path = root_path + Word2Vec_model_path
    Word2Vec_save_path = root_path + Word2Vec_save_path
    model_save_path = root_path + model_save_path
    BERT_save_path = root_path + BERT_save_path
    
    param = {
        'TextCNN': { 
            'sentence_max_length': 300,
            'embed_dim': 200,
            'class_num': 4,
            'label_num': 20,
            "kernel_num": 64,
            "kernel_size": [3, 4, 5],
            "dropout": 0.5,
        },
        'LSTM': {
            'sentence_max_length': 150,
            'embed_dim': 200,
            'class_num': 4,
            'label_num': 20,
            'hidden_size': 256,
            'n_layers': 2,
            "dropout": 0.35
        },
        'BERT': {
            'model_path' : '/home/CodeSpace/Proj1/dataset/other_resources/bert-base-chinese',            # PLM的选择
            'tokenizer_path' : '/home/CodeSpace/Proj1/dataset/other_resources/bert-base-chinese',      # tokenizer的选择，通常与PLM是对应的
            'sentence_max_length': 350,
            'embed_dim': 200,
            'class_num': 4,
            'label_num': 20,
            'dropout': 0.5
        }
    }
    
    train = {
        'TextCNN': {
            'lr': 3e-3,
            'weight_decay': 1e-6,
            'schedule': 'CosineAnnealingLR',
            'epoch': 10,
            'batch_size': 512
        },
        'LSTM': {
            'lr': 0.0005,
            'weight_decay': 1e-6,
            'schedule': 'CosineAnnealingLR',
            'epoch': 30,
            'batch_size': 512
        },
        'BERT': {
            'lr':1e-8,
            'weight_decay': 1e-6,
            'schedule': 'CosineAnnealingLR',
            'epoch': 5,
            'batch_size': 32
        }
    }

    
    