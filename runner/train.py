import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys

sys.path.append("..")
from utils.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from utils.metrics import RMSE_MAE_MAPE
from utils.data_prepare import read_numpy, get_dataloaders, get_dataloaders_from_npz
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
        out_batch = SCALER.inverse_transform(out_batch)
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
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length
    
    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        if cfg["use_cl"]:
            if (
                global_iter_count % cfg["cl_step_size"] == 0
                and global_target_length < cfg["out_steps"]
            ):
                global_target_length += 1
                print_log(f"CL target length = {global_target_length}", log=log)
            loss = criterion(
                out_batch[:, :global_target_length, ...],
                y_batch[:, :global_target_length, ...],
            )
            global_iter_count += 1
        else:
            loss = criterion(out_batch, y_batch)

        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=5,
    max_epochs=500,
    early_stop=10,
    compile_model=False,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
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
def test_model(model, testset_loader, log=None):
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
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--cpus", type=int, default=1)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = args.model.upper()

    with open(f"../config/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    if cfg["pass_device"]:
        cfg["model_args"]["device"] = DEVICE

    model = model_select(model_name)(**cfg["model_args"])

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = f"../logs/{model._get_name()}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model._get_name()}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    print_log(dataset, log=log)
    if cfg["load_npz"]:
        (
            trainset_loader,
            valset_loader,
            testset_loader,
            SCALER,
        ) = get_dataloaders_from_npz(data_path, batch_size=cfg["batch_size"], log=log)
    else:
        if cfg["with_embeddings"]:
            data = read_numpy(
                os.path.join(data_path, f"{dataset}_embedded.npz"), log=log
            )  #!!! (all_steps, num_nodes, 1+time_embedding_dim+node_embedding_dim)
        else:
            data = read_numpy(
                os.path.join(data_path, f"{dataset}.npz"), log=log
            )  # (all_steps, num_nodes)
            data = data[..., np.newaxis]  # (all_steps, num_nodes, 1)
        trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders(
            data,
            cfg["in_steps"],
            cfg["out_steps"],
            batch_size=cfg["batch_size"],
            with_embeddings=cfg["with_embeddings"],
            log=log,
        )
    print_log(log=log)

    save_path = f"../saved_models/{model._get_name()}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model._get_name()}-{dataset}-{now}.pt")

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.1,
    #     patience=15,
    #     min_lr=1e-5,
    #     # threshold=0.01,
    #     # threshold_mode="abs",
    #     threshold=0.05,
    #     threshold_mode="rel",
    #     verbose=True,
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=0.1, verbose=False
    )

    print_log("---------", model._get_name(), "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    summary(
        model,
        [
            cfg["batch_size"],
            cfg["in_steps"],
            cfg["num_nodes"],
            next(iter(trainset_loader))[0].shape[-1],
        ],
    )
    print_log(log=log)
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    if cfg["use_cl"]:
        global_iter_count = 1
        global_target_length = 1
        print_log(f"CL target length = {global_target_length}", log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg["clip_grad"],
        max_epochs=cfg["max_epochs"],
        compile_model=args.compile,
        verbose=1,
        log=log,
        save=save,
    )

    test_model(model, testset_loader, log=log)

    log.close()
