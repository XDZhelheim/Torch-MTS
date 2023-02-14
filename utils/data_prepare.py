import sys
import torch
import pandas as pd
import numpy as np
from .utils import print_log

# ! X shape: (B, T, N, C)


def read_df(data_path, file_type="pickle", transpose=False, log="train.log"):
    """
    Returns
    ---
    X: (all_timesteps, num_nodes) numpy
    """
    if file_type == "pickle":
        data = pd.read_pickle(data_path)
    elif file_type == "csv":
        data = pd.read_csv(data_path)
    else:
        print("Invalid file type.")
        sys.exit(1)

    data = data.values.astype(np.float)
    if transpose:
        data = data.T
    print_log("Original data shape", data.shape, log=log)
    return data


def read_numpy(data_path, transpose=False, log="train.log"):
    """
    Returns
    ---
    X: (all_timesteps, num_nodes) numpy
    """

    data = np.load(data_path)
    if transpose:
        data = data.T
    print_log("Original data shape", data.shape, log=log)
    return data


def gen_xy(data, in_steps, out_steps, with_time_embeddings=False):
    """
    Parameter
    ---
    data: (all_timesteps, num_nodes, 1+time_embedding_dim) if with_time_embeddings, else features=1

    Returns
    ---
    x: (num_samples, in_steps, num_nodes, num_features=1+time_embedding_dim or 1) Tensor
    y: (num_samples, out_steps, num_nodes, num_features=1) Tensor
        num_samples is determined by `timesteps` and `in_steps+out_steps`
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    all_steps = data.shape[0]
    indices = [
        (i, i + (in_steps + out_steps))
        for i in range(all_steps - (in_steps + out_steps) + 1)
    ]

    x, y = [], []
    for begin, end in indices:
        x.append(data[begin : begin + in_steps])
        y.append(data[begin + in_steps : end])

    x = np.array(x)
    y = np.array(y)

    if with_time_embeddings:
        y = y[..., 0][..., np.newaxis]

    return torch.Tensor(x), torch.Tensor(y)


def get_dataloaders(
    data,
    in_steps,
    out_steps,
    train_size=0.7,
    val_size=0.1,
    batch_size=32,
    with_time_embeddings=False,
    num_cpu=8,
    log="train.log",
):
    """
    Parameters
    ---
    data: (all_timesteps, num_nodes, 1+time_embedding_dim or 1) numpy
    """
    all_steps = data.shape[0]
    split1 = int(all_steps * train_size)
    split2 = int(all_steps * (train_size + val_size))

    train_data = data[:split1]
    val_data = data[split1:split2]
    test_data = data[split2:]

    x_train, y_train = gen_xy(train_data, in_steps, out_steps, with_time_embeddings)
    x_val, y_val = gen_xy(val_data, in_steps, out_steps, with_time_embeddings)
    x_test, y_test = gen_xy(test_data, in_steps, out_steps, with_time_embeddings)

    print_log(f"Trainset:\tx-{x_train.size()}\ty-{y_train.size()}", log=log)
    print_log(f"Valset:  \tx-{x_val.size()}  \ty-{y_val.size()}", log=log)
    print_log(f"Testset:\tx-{x_test.size()}\ty-{y_test.size()}", log=log)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_cpu
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True, num_workers=num_cpu
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_cpu
    )

    return trainset_loader, valset_loader, testset_loader
