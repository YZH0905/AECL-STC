"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

BERT_CLASS = {'distilroberta': 'distilbert-base-uncased'}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        print(args.bert)
        print(BERT_CLASS[args.bert])
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])                  # 加载模型配置（参数可以为模型名称，也可以为具体文件）
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)     # 加载预训练模型（参数可以为模型名称，也可以为具体文件）
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])            # 加载分词器  （参数可以为模型名称，也可以为具体文件）
        print("..... loading plain BERT !!!")
        
    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer('pretrain-model')
    # sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert








