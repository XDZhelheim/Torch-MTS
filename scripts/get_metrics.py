import os
import re
import sys

sys.path.append("..")
from lib.utils import print_log

log_path = "../logs/"


def get_metrics_log(log: str):
    with open(log, "r") as f:
        lines = f.readlines()

    metrics = []
    for line in lines:
        if (
            line.startswith("All Steps")
            or line.startswith("Step 3")
            or line.startswith("Step 6")
            or line.startswith("Step 12")
        ):
            value_list = list(map(float, re.findall("\d+\.?\d*", line)))

            value_list[-1] = value_list[-1] / 100  # MAPE/100

            if len(value_list) == 5:
                value_list.pop(0)
                value_list[0] = 0

            metrics.append(value_list)

    metrics.append(metrics.pop(0))  # Avg in the end

    return metrics


def print_model_metrics(model: str, dataset=None, file=None):
    model_logs = os.path.join(log_path, model)
    for log in sorted(os.listdir(model_logs)):
        if dataset:
            if model not in log or dataset.upper() not in log:
                continue

        print_log(log, log=file)
        for line in get_metrics_log(os.path.join(model_logs, log)):
            for value in line:
                if value % 1 == 0:
                    print_log(int(value), end="\t", log=file)
                else:
                    print_log("%.4f" % value, end="\t\t", log=file)
            print_log(log=file)
        print_log(log=file)


def print_model_metrics_csv(models, datasets, file=None):
    print_log("Dataset,Model,Step,MAE,RMSE,MAPE", log=file)

    for dataset in datasets:
        for model in models:
            model_logs = os.path.join(log_path, model)
            for log in sorted(os.listdir(model_logs)):
                if dataset:
                    if model not in log or dataset.upper() not in log:
                        continue

                for line in get_metrics_log(os.path.join(model_logs, log)):
                    print_log(
                        f"{dataset.upper()},{model},{int(line[0])},{line[1]:.4f},{line[2]:.4f},{line[3]:.4f}",
                        log=file,
                    )
                print_log(log=file)


def print_model_metrics_csv_long(models, datasets, file=None):
    print_log(
        "Dataset,Model,MAE_3,RMSE_3,MAPE_3,MAE_6,RMSE_6,MAPE_6,MAE_12,RMSE_12,MAPE_12,MAE_all,RMSE_all,MAPE_all",
        log=file,
    )

    for dataset in datasets:
        for model in models:
            model_logs = os.path.join(log_path, model)
            for log in sorted(os.listdir(model_logs)):
                if dataset:
                    if model not in log or dataset.upper() not in log:
                        continue

                print_log(f"{dataset.upper()},{model}", end=",", log=file)
                for line in get_metrics_log(os.path.join(model_logs, log)):
                    print_log(
                        f"{line[1]:.4f},{line[2]:.4f},{line[3]:.4f}",
                        end="," if line[0] else "\n",
                        log=file,
                    )  # if line[0]==0


if __name__ == "__main__":
    models = [
        "HistoricalInertia",
        "MLP",
        "LSTM",
        "GRU",
        "WaveNet",
        "Transformer",
        "Mamba",
        "GCLSTM",
        "GCGRU",
        "STGCN",
        "DCRNN",
        "AGCRN",
        "GWNET",
        "MTGNN",
        "StemGNN",
        "STNorm",
        "GTS",
        "STID",
        "STWA",
        "MegaCRN",
        "STAEformer",
    ]
    datasets = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMSD7M", "PEMSD7L"]

    for dataset in datasets:
        for model in models:
            print_model_metrics(model, dataset)

    # file = open(os.path.join(log_path, "results.csv"), "a")
    # file.seek(0)
    # file.truncate()
    # print_model_metrics_csv(models, datasets, file=file)
    
    # file = open(os.path.join(log_path, "results_long.csv"), "a")
    # file.seek(0)
    # file.truncate()
    # print_model_metrics_csv_long(models, datasets, file=file)
