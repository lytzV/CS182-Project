from model import YelpModel
from data_pipeline import YelpDataset
from trainer import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


if __name__ == '__main__':
    TRAIN_VAL_SPLIT = 0.8
    hparams = {
        "batch_size":20,
        "learning_rate":1e-4,
        "num_epoch":1
    }
    model_params = {}


    dataset = YelpDataset("yelp_review_training_dataset.jsonl", "sentence-transformers/bert-base-nli-stsb-mean-tokens")
    len_dataset = len(dataset)
    train_dataset, val_dataset = random_split(dataset, [TRAIN_VAL_SPLIT*len_dataset, (1-TRAIN_VAL_SPLIT)*len_dataset])
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], collate_fn=lambda x: train_dataset.data_processing(x))
    val_dataloader = DataLoader(val_dataset, batch_size=hparams['batch_size'], collate_fn=lambda x: val_dataset.data_processing(x))

    model = YelpModel(model_params)
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    loss = nn.CrossEntropyLoss()

    train(model, train_dataloader, optimizer, loss, hparams['num_epoch'])
