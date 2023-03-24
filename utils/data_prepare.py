import sys
import torch
import pandas as pd
import numpy as np
import os
from .utils import print_log, StandardScaler

# ! X shape: (B, T, N, C)


def read_df(data_path, file_type="pickle", transpose=False, log=None):
    """
    Returns
    ---
    X: (all_timesteps, num_nodes) numpy
    """
    if file_type == "pickle":
        data = pd.read_pickle(data_path)
    elif file_type == "csv":
        data = pd.read_csv(data_path)
    elif file_type == "h5":
        data = pd.read_hdf(data_path)
    else:
        raise TypeError("Unsupported file type.")

    data = data.values.astype(np.float32)
    if transpose:
        data = data.T
    print_log("Original data shape", data.shape, log=log)
    return data


def read_numpy(data_path, transpose=False, log=None):
    """
    Returns
    ---
    X: (all_timesteps, num_nodes) numpy
    """

    if data_path.endswith("npy"):
        data = np.load(data_path).astype(np.float32)
    elif data_path.endswith("npz"):
        data = np.load(data_path)["data"].astype(np.float32)
    else:
        raise TypeError("Unsupported file type.")
    if transpose:
        data = data.T
    print_log("Original data shape", data.shape, log=log)
    return data


def gen_xy(data, in_steps, out_steps, with_embeddings=False):
    """
    Parameter
    ---
    data: (all_timesteps, num_nodes, 1+embedding_dim) if with_embeddings, else features=1

    Returns
    ---
    x: (num_samples, in_steps, num_nodes, num_features=1+time_embedding_dim or 1) Numpy
    y: (num_samples, out_steps, num_nodes, num_features=1) Numpy
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
        if with_embeddings:
            y.append(data[begin + in_steps : end, :, 0])
        else:
            y.append(data[begin + in_steps : end])

    x = np.array(x)
    y = np.array(y)

    if with_embeddings:
        y = y[..., np.newaxis]

    return x, y


def get_dataloaders(
    data,
    in_steps,
    out_steps,
    train_size=0.7,
    val_size=0.1,
    batch_size=32,
    with_embeddings=False,
    log=None,
):
    """
    Parameters
    ---
    data: (all_timesteps, num_nodes, 1+time_embedding_dim or 1) numpy
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.shape[2] > 1:
        data = data[..., :1]  # for PEMS04 and 08, only use traffic flow

    all_steps = data.shape[0]
    split1 = int(all_steps * train_size)
    split2 = int(all_steps * (train_size + val_size))

    train_data = data[:split1]
    val_data = data[split1:split2]
    test_data = data[split2:]

    x_train, y_train = gen_xy(train_data, in_steps, out_steps, with_embeddings)
    x_val, y_val = gen_xy(val_data, in_steps, out_steps, with_embeddings)
    x_test, y_test = gen_xy(test_data, in_steps, out_steps, with_embeddings)

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    # ! do not transform y
    # y_train[..., 0] = scaler.transform(y_train[..., 0])
    # y_val[..., 0] = scaler.transform(y_val[..., 0])
    # y_test[..., 0] = scaler.transform(y_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler


def get_dataloaders_from_npz(
    data_path, batch_size=32, log=None,
):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(data_path, category + ".npz"))
        data["x_" + category] = cat_data["x"].astype(np.float32)
        data["y_" + category] = cat_data["y"].astype(np.float32)

    print_log(
        f"Trainset:\tx-{data['x_train'].shape}\ty-{data['y_train'].shape}", log=log
    )
    print_log(f"Valset:  \tx-{data['x_val'].shape}  \ty-{data['y_val'].shape}", log=log)
    print_log(f"Testset:\tx-{data['x_test'].shape}\ty-{data['y_test'].shape}", log=log)

    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
        # data["y_" + category][..., 0] = scaler.transform(data["y_" + category][..., 0])

    for category in ["train", "val", "test"]:
        data["x_" + category] = torch.FloatTensor(data["x_" + category])
        data["y_" + category] = torch.FloatTensor(
            data["y_" + category][..., :1]
        )  # no time embedding

    trainset = torch.utils.data.TensorDataset(data["x_train"], data["y_train"])
    valset = torch.utils.data.TensorDataset(data["x_val"], data["y_val"])
    testset = torch.utils.data.TensorDataset(data["x_test"], data["y_test"])

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
