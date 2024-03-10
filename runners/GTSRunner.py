from .AbstractRunner import AbstractRunner

import torch
import numpy as np
from torchinfo import summary


class GTSRunner(AbstractRunner):
    def __init__(
        self,
        cfg: dict,
        device,
        scaler,
        log=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = cfg.get("clip_grad")

        self.batches_seen = 0

    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion):
        model.train()
        batch_loss_list = []
        for x_batch, y_batch in trainset_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output, pred_adj, prior_adj = model(x_batch, self.scaler.transform(y_batch), self.batches_seen) # !!! important: transform y_true
            y_pred = self.scaler.inverse_transform(output)
            
            # What form of power is this??
            # https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_supervisor.py#L191
            # Reason: parameters are created at the first iteration
            if self.batches_seen == 0:
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.cfg.get("lr", 0.01),
                    weight_decay=self.cfg.get("weight_decay", 0),
                    eps=self.cfg.get("eps", 0.001),
                )
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=self.cfg.get("milestones", []),
                    gamma=self.cfg.get("lr_decay_rate", 0.1),
                    verbose=False,
                )

            self.batches_seen += 1

            loss = criterion(y_pred, y_batch, pred_adj, prior_adj)
            batch_loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            self.optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        self.scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, valset_loader, criterion):
        model.eval()
        batch_loss_list = []
        for x_batch, y_batch in valset_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output, pred_adj, prior_adj = model(x_batch)
            y_pred = self.scaler.inverse_transform(output)

            loss = criterion(y_pred, y_batch, pred_adj, prior_adj)
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y_list = []
        out_list = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output, _, _ = model(x_batch)
            y_pred = self.scaler.inverse_transform(output)

            y_pred = y_pred.cpu().numpy()
            y_true = y_batch.cpu().numpy()
            out_list.append(y_pred)
            y_list.append(y_true)

        out = np.vstack(out_list)
        y = np.vstack(y_list)

        return y, out

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape

        return summary(
            model,
            x_shape,
            verbose=0,  # avoid print twice
            device=self.device,
        )
