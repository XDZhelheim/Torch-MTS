import numpy as np
import torch
import torch.nn as nn


def loss_select(name):
    name = name.upper()

    # default loss of each dataset
    if name in ("METRLA", "PEMSBAY", "PEMSD7M", "PEMSD7L"):
        return MaskedMAELoss
    elif name in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        return nn.HuberLoss
    elif name in (
        "ELECTRICITY",
        "EXCHANGE",
        "TRAFFIC",
        "WEATHER",
        "ILI",
        "ETTH1",
        "ETTH2",
        "ETTM1",
        "ETTM2",
    ):
        return nn.MSELoss

    elif name in (
        "MASKEDMAELOSS",
        "MASKED_MAE_LOSS",
        "MASKEDMAE",
        "MASKED_MAE",
        "MASKMAE",
        "MASK_MAE",
        "MMAE",
    ):
        return MaskedMAELoss
    elif name in (
        "HUBERLOSS",
        "HUBER_LOSS",
        "HUBER",
        "SMOOTHEDL1LOSS",
        "SMOOTHED_L1_LOSS",
        "SMOOTHEDL1",
        "SMOOTHED_L1",
    ):
        return nn.HuberLoss
    elif name in ("MAELOSS", "MAE_LOSS", "MAE", "L1LOSS", "L1_LOSS", "L1"):
        return nn.L1Loss
    elif name in ("MSELOSS", "MSE_LOSS", "MSE"):
        return nn.MSELoss

    elif name in ("MEGACRNLOSS", "MEGACRN"):
        return MegaCRNLoss
    elif name in ("GTSLoss", "GTS"):
        return GTSLoss

    else:
        raise NotImplementedError


def masked_mae_loss(preds, labels, null_val=0.0):
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


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def masked_mae_loss_vDCRNN(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


class MaskedMAELoss_vDCRNN:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, y_pred, y_true):
        return masked_mae_loss_vDCRNN(y_pred, y_true)


class MegaCRNLoss:
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2
        self.masked_mae_loss = MaskedMAELoss()
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()

    def _get_name(self):
        return self.__class__.__name__

    def forward(self, y_pred, y_true, query, pos, neg):
        loss1 = self.masked_mae_loss(y_pred, y_true)
        loss2 = self.separate_loss(query, pos.detach(), neg.detach())
        loss3 = self.compact_loss(query, pos.detach())

        loss = loss1 + self.l1 * loss2 + self.l2 * loss3

        return loss

    def __call__(self, y_pred, y_true, query, pos, neg):
        return self.forward(y_pred, y_true, query, pos, neg)


class GTSLoss:
    def __init__(self):
        self.masked_mae_loss = MaskedMAELoss()
        self.graph_loss = nn.BCELoss()

    def _get_name(self):
        return self.__class__.__name__

    def forward(self, y_pred, y_true, pred_adj, prior_adj):
        # graph loss
        prior_label = prior_adj.view(prior_adj.shape[0] * prior_adj.shape[1]).to(
            pred_adj.device
        )
        pred_label = pred_adj.view(pred_adj.shape[0] * pred_adj.shape[1])
        loss_g = self.graph_loss(pred_label, prior_label)

        # regression loss
        loss_r = self.masked_mae_loss(y_pred, y_true)

        # total loss
        loss = loss_r + loss_g
        return loss

    def __call__(self, y_pred, y_true, pred_adj, prior_adj):
        return self.forward(y_pred, y_true, pred_adj, prior_adj)
