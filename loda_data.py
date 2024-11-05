import os
import torch
from torch.utils import data
import pandas as pd
from collections import Counter

class test_Dataset(data.Dataset):
    def __init__(self, train_x0, label, tokenizer, args):
        assert len(train_x0) == len(label)
        self.train_x0 = train_x0
        self.label = label
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        feat0 = self.tokenizer.encode_plus(
            self.train_x0[idx],
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = torch.cat([feat0['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat0['attention_mask'].unsqueeze(1)], dim=1)

        return input_ids, attention_mask, self.label[idx], idx


class train_Dataset(data.Dataset):
    def __init__(self, train_x1, train_x2, tokenizer, args):
        assert len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_x2)

    def __getitem__(self, idx):
        feat1 = self.tokenizer.encode_plus(
            self.train_x1[idx],
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        feat2 = self.tokenizer.encode_plus(
            self.train_x2[idx],
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)

        return input_ids, attention_mask, idx


def explict_augmentation_loader(data_name):
    data = pd.read_csv("./prepare_data/augmented data/" + data_name + ".csv")
    text0 = data["text0"].fillna('.').values
    text1 = data["text1"].fillna('.').values
    label = data["label"].astype(int).values

    return text0, text1, label