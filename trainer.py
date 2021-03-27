import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch.nn import functional as F

def train(model, training_data, optimizer, loss, num_epoch):
    for e in range(num_epoch):
        print('Epoch: ' + e)
        for batch_idx, data in enumerate(training_data):
            print('Batch: ' + batch_idx)
            input_ids, attention_mask, labels = data

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask = attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()
