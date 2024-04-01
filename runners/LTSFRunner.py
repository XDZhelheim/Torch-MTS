from .AbstractRunner import AbstractRunner

import torch
import numpy as np
from torchinfo import summary
import sys
import datetime
import copy
import time
import matplotlib.pyplot as plt

sys.path.append("..")
from lib.utils import print_log
from lib.metrics import MSE_MAE


class LTSFRunner(AbstractRunner):
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

    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion):
        model.train()
        batch_loss_list = []
        for x_batch, y_batch in trainset_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = model(x_batch)

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

            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            out.append(out_batch)
            y.append(y_batch)

        out = np.vstack(out)  # (samples, out_steps, num_nodes, output_dim)
        y = np.vstack(y)

        return y, out

    def train(
        self,
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=200,
        early_stop=10,
        compile_model=False,
        verbose=1,
        plot=False,
        save=None,
    ):
        if torch.__version__ >= "2.0.0" and compile_model:
            model = torch.compile(model)

        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        for epoch in range(max_epochs):
            train_loss = self.train_one_epoch(
                model, trainset_loader, optimizer, scheduler, criterion
            )
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(model, valset_loader, criterion)
            val_loss_list.append(val_loss)

            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    " \tTrain Loss = %.5f" % train_loss,
                    "Val Loss = %.5f" % val_loss,
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= early_stop:
                    break

        model.load_state_dict(best_state_dict)
        train_mse, train_mae = MSE_MAE(*self.predict(model, trainset_loader))
        val_mse, val_mae= MSE_MAE(*self.predict(model, valset_loader))

        out_str = f"Early stopping at epoch: {epoch+1}\n"
        out_str += f"Best at epoch {best_epoch+1}:\n"
        out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
        out_str += "Train MSE = %.5f, MAE = %.5f\n" % (
            train_mse,
            train_mae,
        )
        out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
        out_str += "Val MSE = %.5f, MAE = %.5f" % (
            val_mse,
            val_mae,
        )
        print_log(out_str, log=self.log)

        if plot:
            plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
            plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
            plt.title("Epoch-Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        if save:
            torch.save(best_state_dict, save)
        return model

    @torch.no_grad()
    def test_model(self, model, testset_loader):
        model.eval()
        print_log("--------- Test ---------", log=self.log)

        start = time.time()
        y_true, y_pred = self.predict(model, testset_loader)
        end = time.time()

        out_steps = y_pred.shape[1]

        mse_all, mae_all = MSE_MAE(y_true, y_pred)
        out_str = "All Steps (1-%d) MSE = %.5f, MAE = %.5f\n" % (
            out_steps,
            mse_all,
            mae_all,
        )

        print_log(out_str, log=self.log, end="")
        print_log("Inference time: %.2f s" % (end - start), log=self.log)

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape

        return summary(
            model,
            x_shape,
            verbose=0,  # avoid print twice
            device=self.device,
        )
