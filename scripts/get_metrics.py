import os
import re

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

            value_list[-3], value_list[-2] = value_list[-2], value_list[-3]  # MAE RMSE
            value_list[-1] = value_list[-1] / 100  # MAPE/100

            if len(value_list) == 3:
                value_list.insert(0, -1)

            metrics.append(value_list)

    metrics.append(metrics.pop(0))  # Avg in the end

    return metrics


def print_model_metrics(model: str, dataset=None):
    model_logs = os.path.join(log_path, model)
    for log in sorted(os.listdir(model_logs)):
        if dataset:
            if dataset.upper() not in log:
                continue

        print(log)
        for line in get_metrics_log(os.path.join(model_logs, log)):
            for value in line:
                if value % 1 == 0:
                    print(int(value), end="\t")
                else:
                    print("%.4f" % value, end="\t\t")
            print()
        print()


if __name__ == "__main__":
    model = "LSTM"
    datasets = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]

    for dataset in datasets:
        print_model_metrics(model, dataset)
