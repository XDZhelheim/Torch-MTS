import numpy as np
import torch

class StandardScaler():
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """
    
    # def __init__(self, mean, std):
    #     self.mean = mean
    #     self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
        
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    # preds[preds < 1e-5] = 0
    # labels[labels < 1e-5] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def onehot_decode(label):
    return torch.argmax(label, dim=1)
