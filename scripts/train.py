import argparse
import os
import torch
import datetime
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
from lib.data_prepare import dataloader_select
from lib.losses import loss_select
from models import model_select
from runners import runner_select

# ! X shape: (B, T, N, C)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="METRLA")
    parser.add_argument("-m", "--model", type=str, default="LSTM")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=233, help="Set random seed")
    parser.add_argument("--cpus", type=int, default=1, help="Limit number of cpu threads used")
    parser.add_argument("--config", type=str, default=None, help="Specify .yaml config file path")

    parser.add_argument("-s", "--seq_len", type=int, default=0, help="seq_len for LTSF")
    parser.add_argument("-p", "--pred_len", type=int, default=0, help="pred_len for LTSF")
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

    cfg_path = args.config or f"../configs/{model_name}.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # shortcut for LTSF
    if args.seq_len > 0:
        assert "seq_len" in cfg["model_args"], "Specifying input length is only for LTSF."
        cfg["in_steps"] = args.seq_len
        cfg["model_args"]["seq_len"] = args.seq_len
    if args.pred_len > 0:
        assert "pred_len" in cfg["model_args"], "Specifying prediction length is only for LTSF."
        cfg["out_steps"] = args.pred_len
        cfg["model_args"]["pred_len"] = args.pred_len

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
    ) = dataloader_select(cfg.get("dataloader", dataset))(
        data_path,
        in_steps=cfg.get("in_steps", 12),
        out_steps=cfg.get("out_steps", 12),
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

    runner = runner_select(cfg.get("runner", "STF"))(
        cfg, device=DEVICE, scaler=SCALER, log=log
    )

    # --------------------------- print model structure -------------------------- #

    print_log(f"Random seed = {args.seed}", log=log)
    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(runner.model_summary(model, trainset_loader), log=log)
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = runner.train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        compile_model=args.compile,
        verbose=1,
        save=save,
    )

    print_log(f"Model checkpoint saved to: {save}", log=log)

    runner.test_model(model, testset_loader)

    log.close()
