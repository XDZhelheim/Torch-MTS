import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torchinfo import summary
import yaml
import pandas as pd

import sys

sys.path.append("..")
from utils.utils import masked_mae_loss, print_log
from utils.metrics import RMSE_MAE_MAPE
from utils.data_prepare import read_df, read_numpy, get_dataloaders
from model import model_select

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    out = SCALER.inverse_transform(out)
    y = SCALER.inverse_transform(y)  # (samples, out_steps, num_nodes)

    return y, out


def train_one_epoch(model, trainset_loader, optimizer, criterion):
    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(batch_loss_list)


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    max_epochs=500,
    early_stop=10,
    verbose=1,
    plot=False,
    log="train.log",
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, trainset_loader, optimizer, criterion)
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

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
def test_model(model, testset_loader, log="train.log"):
    model.eval()
    print_log("--------- Test ---------", log=log)
    y_true, y_pred = predict(model, testset_loader)
    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="METRLA")
    parser.add_argument("-m", "--model", type=str, default="LSTM")
    parser.add_argument("-g", "--gpu_num", type=int, default=1)
    args = parser.parse_args()

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    DATA_PATH = f"../data/{dataset}"
    
    with open(f"../config/{args.model}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    if cfg["load_adj"]:
        adj = pd.read_csv(cfg["adj_path"]).values
        cfg["model_args"]["adj"] = adj
        cfg["model_args"]["adj_device"] = DEVICE

    model = model_select(args.model)(**cfg["model_args"])

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = f"../logs/{model._get_name()}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model._get_name()}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    print_log(dataset, log=log)
    SCALER = StandardScaler()
    data = read_numpy(
        os.path.join(DATA_PATH, f"{dataset}.npy"), log=log
    )  # (all_steps, num_nodes)
    data = SCALER.fit_transform(data)
    data = data[:, :, np.newaxis]  # (all_steps, num_nodes, 1)
    trainset_loader, valset_loader, testset_loader = get_dataloaders(
        data,
        cfg["in_steps"],
        cfg["out_steps"],
        batch_size=cfg["batch_size"],
        with_time_embeddings=False,
        num_cpu=8,
        log=log,
    )
    print_log(log=log)

    save_path = f"../saved_models/{model._get_name()}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model._get_name()}-{dataset}-{now}.pt")

    if dataset == "METRLA":
        criterion = masked_mae_loss
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    
    print_log("---------", model._get_name(), "---------", log=log)
    print_log(cfg["model_args"], log=log)
    summary(model, [cfg["batch_size"], cfg["in_steps"], cfg["num_nodes"], 1])
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        criterion,
        max_epochs=cfg["max_epochs"],
        verbose=1,
        log=log,
        save=save,
    )

    test_model(model, testset_loader, log=log)

    log.close()
