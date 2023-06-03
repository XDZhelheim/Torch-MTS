import argparse
import numpy as np
import os
import torch
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys

sys.path.append("..")
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from lib.losses import loss_select
from models import model_select
from runners import runner_select

# ! X shape: (B, T, N, C)


def train(
    model,
    runner,
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
    log=None,
    save=None,
):
    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = runner.train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion
        )
        train_loss_list.append(train_loss)

        val_loss = runner.eval_model(model, valset_loader, criterion)
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
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(
        *runner.predict(model, trainset_loader)
    )
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*runner.predict(model, valset_loader))

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
def test_model(model, runner, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = runner.predict(model, testset_loader)
    end = time.time()

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
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="METRLA")
    parser.add_argument("-m", "--model", type=str, default="LSTM")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
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

    model_class = model_select(model_name)
    model_name = model_class.__name__

    with open(f"../configs/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    # cfg.get(key, default_value=None): no need to write in the config if not used
    # cfg[key]: must be assigned in the config, else KeyError
    if cfg.get("pass_device"):
        cfg["model_args"]["device"] = DEVICE

    model = model_class(**cfg["model_args"]).to(DEVICE)

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/{model_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        y_tod=cfg.get("y_time_of_day"),
        y_dow=cfg.get("y_day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = loss_select(cfg.get("loss", dataset))(**cfg.get("loss_args", {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get("milestones", []),
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # ----------------------------- set model runner ----------------------------- #

    runner = runner_select(cfg.get("runner", "basic"))(
        cfg, device=DEVICE, scaler=SCALER, log=log
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(runner.model_summary(model, trainset_loader), log=log)
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        runner,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        compile_model=args.compile,
        verbose=1,
        log=log,
        save=save,
    )

    test_model(model, runner, testset_loader, log=log)

    log.close()
