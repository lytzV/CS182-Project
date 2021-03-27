import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

class YelpDataset(Dataset):
    def __init__(self, file_path, pretrained_model):
        self.file_path = file_path
        self.data = pd.read_json(file_path,lines=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['stars']
        text = row['text']
        return {"text":text, "label":label}
    def data_processing(self, data):
        text_batch = []
        label_batch = []
        for d in data:
            text_batch.append(d['text'])
            label_batch.append(d['label'])
        encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids, attn_masks = encoding['input_ids'], encoding['attention_mask']
        labels = torch.tensor(label_batch)
        return input_ids, attn_masks, labels
