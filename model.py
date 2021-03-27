import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from transformers import BertForSequenceClassification


class YelpModel(nn.Module):
    def __init__(self, model_params):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model.train()

    def forward(self, data):
        input_ids, attention_mask, labels = data
        self.model(input_ids, attention_mask = attention_mask)
