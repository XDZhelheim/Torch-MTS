from .AbstractRunner import AbstractRunner

import torch
import numpy as np
from torchinfo import summary
import sys

sys.path.append("..")
from lib.utils import print_log


class BasicRunner(AbstractRunner):
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

        if cfg.get("use_cl"):
            if "cl_step_size" not in cfg:
                raise KeyError("Missing config: cl_step_size (int).")
            if "out_steps" not in cfg:
                raise KeyError("Missing config: out_steps (int).")
            self.iter_count = 0
            self.target_length = 0

        self.clip_grad = cfg.get("clip_grad")

    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion):
        model.train()
        batch_loss_list = []
        for x_batch, y_batch in trainset_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)

            if self.cfg.get("use_cl"):
                if (
                    self.iter_count % self.cfg["cl_step_size"] == 0
                    and self.target_length < self.cfg["out_steps"]
                ):
                    self.target_length += 1
                    print_log(f"CL target length = {self.target_length}", log=self.log)
                loss = criterion(
                    out_batch[:, : self.target_length, ...],
                    y_batch[:, : self.target_length, ...],
                )
                self.iter_count += 1
            else:
                loss = criterion(out_batch, y_batch)

            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, valset_loader, criterion):
        model.eval()
        batch_loss_list = []
        for x_batch, y_batch in valset_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)
            loss = criterion(out_batch, y_batch)
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y = []
        out = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)

            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            out.append(out_batch)
            y.append(y_batch)

        out = np.vstack(out)  # (samples, out_steps, num_nodes, output_dim)
        y = np.vstack(y)

        return y, out

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape

        return summary(
            model,
            x_shape,
            verbose=0,  # avoid print twice
            device=self.device,
        )
