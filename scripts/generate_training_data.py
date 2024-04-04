"""
This file is adapted from BasicTS:

https://github.com/zezhishao/BasicTS/blob/master/scripts/data_preparation/METR-LA/generate_training_data.py
https://github.com/zezhishao/BasicTS/blob/master/scripts/data_preparation/PEMS03/generate_training_data.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

def generate_data(
    dataset_dir,
    data_file_path,
    target_channel=[0],
    history_seq_len=12,
    future_seq_len=12,
    add_time_of_day=True,
    add_day_of_week=True,
    train_ratio=0.7,
    valid_ratio=0.1,
    steps_per_day=288,
    date_format="%Y-%m-%d %H:%M:%S",
    save_data=True,
    split_first=False,
):
    """Preprocess and generate train/valid/test datasets.
    
    Default settings of METRLA and PEMSBAY dataset:
        - Dataset division: 7:1:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic speed, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.
        
    Default settings of PEMS03/04/07/08 dataset:
        - Dataset division: 6:2:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic flow, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.
    """
    
    if data_file_path.endswith("h5"):
        file_type = "hdf"
        df = pd.read_hdf(data_file_path)
        data = np.expand_dims(df.values, axis=-1)
    elif data_file_path.endswith("npz"):
        file_type = "npz"
        data = np.load(data_file_path)["data"]
    elif data_file_path.endswith("csv"):
        file_type = "csv"
        df = pd.read_csv(data_file_path)
        df_index = pd.to_datetime(df["date"].values, format=date_format).to_numpy()
        df = df[df.columns[1:]]
        df.index = df_index
        data = np.expand_dims(df.values, axis=-1)
    else:
        raise TypeError("Unsupported file type.")

    data = data[..., target_channel] # (all_steps, num_nodes, num_channels)
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    if split_first:
        # first split train/val/test, then perform sliding window individually
        split1 = round(l * train_ratio)
        split2 = round(l * (train_ratio + valid_ratio))
        train_index = [(t - history_seq_len, t, t + future_seq_len) for t in range(history_seq_len, split1 - future_seq_len + 1)]
        valid_index = [(t - history_seq_len, t, t + future_seq_len) for t in range(split1 + history_seq_len, split2 - future_seq_len + 1)]
        test_index = [(t - history_seq_len, t, t + future_seq_len) for t in range(split2 + history_seq_len, l - future_seq_len + 1)]
    else:
        # first sliding window, then split (default setting)
        # commonly used for spatiotemporal/traffic forecasting datasets
        # actually this is not strict because it will cross the boundaries of train&val, val&test
        # will generate more samples than split_first
        num_samples = l - (history_seq_len + future_seq_len) + 1
        train_num_short = round(num_samples * train_ratio)
        valid_num_short = round(num_samples * valid_ratio)
        test_num_short = num_samples - train_num_short - valid_num_short

        index_list = [(t - history_seq_len, t, t + future_seq_len) for t in range(history_seq_len, num_samples + history_seq_len)]

        train_index = index_list[:train_num_short]
        valid_index = index_list[train_num_short : train_num_short + valid_num_short]
        test_index = index_list[train_num_short +
                                valid_num_short : train_num_short + valid_num_short + test_num_short]
        
    print("number of training samples: {0}".format(len(train_index)))
    print("number of validation samples: {0}".format(len(valid_index)))
    print("number of test samples: {0}".format(len(test_index)))

    # add external feature
    feature_list = [data]
    if add_time_of_day:
        # numerical time_of_day
        if file_type == "hdf" or file_type == "csv":
            tod = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        elif file_type == "npz":
            tod = np.array([i % steps_per_day / steps_per_day for i in range(data.shape[0])])
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        if file_type == "hdf" or file_type == "csv":
            dow = df.index.dayofweek
        elif file_type == "npz":
            dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1) # (all_steps, num_nodes, num_channels+tod+dow)
    print("data shape: {0}".format(processed_data.shape))

    # dump data
    np.savez_compressed(os.path.join(dataset_dir, f"index_{history_seq_len}_{future_seq_len}.npz"), train=train_index, val=valid_index, test=test_index)
    if save_data:
        np.savez_compressed(os.path.join(dataset_dir, f"data.npz"), data=processed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, 
                        default="METRLA", help="Which dataset to run")
    parser.add_argument("-p", "--history_seq_len", type=int,
                        default=12, help="History sequence length.")
    parser.add_argument("-f", "--future_seq_len", type=int,
                        default=12, help="Future sequence length.")
    parser.add_argument("--target_channel", type=list,
                        default=[0], help="Selected channels.")
    # parser.add_argument("--tod", action="store_true",
    #                     help="Add feature time_of_day.")
    # parser.add_argument("--dow", action="store_true",
    #                     help="Add feature day_of_week.")
    # parser.add_argument("--train_ratio", type=float,
    #                     default=False, help="Train ratio")
    # parser.add_argument("--valid_ratio", type=float,
    #                     default=False, help="Validate ratio.")
    args = parser.parse_args()
    
    DATASET_NAME = args.dataset.upper()
    
    param_dict = {}
    param_dict["history_seq_len"] = args.history_seq_len
    param_dict["future_seq_len"] = args.future_seq_len
    param_dict["target_channel"] = args.target_channel # target channel(s)
    param_dict["add_time_of_day"] = True # if add time_of_day feature
    param_dict["add_day_of_week"] = True # if add day_of_week feature
    param_dict["dataset_dir"] = os.path.join("../data/", DATASET_NAME)
    if DATASET_NAME in ("METRLA", "PEMSBAY"):
        param_dict["data_file_path"] = os.path.join("../data/", DATASET_NAME, f"{DATASET_NAME}.h5")
        param_dict["train_ratio"] = 0.7
        param_dict["valid_ratio"] = 0.1
    elif DATASET_NAME in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        param_dict["data_file_path"] = os.path.join("../data/", DATASET_NAME, f"{DATASET_NAME}.npz")
        param_dict["train_ratio"] = 0.6
        param_dict["valid_ratio"] = 0.2
        param_dict["steps_per_day"] = 288
    elif DATASET_NAME in ("ELECTRICITY", "WEATHER", "TRAFFIC", "ILI"):
        param_dict["data_file_path"] = os.path.join("../data/", DATASET_NAME, f"{DATASET_NAME}.csv")
        param_dict["train_ratio"] = 0.7
        param_dict["valid_ratio"] = 0.1
    elif DATASET_NAME == "EXCHANGE":
        param_dict["data_file_path"] = os.path.join("../data/", DATASET_NAME, f"{DATASET_NAME}.csv")
        param_dict["train_ratio"] = 0.7
        param_dict["valid_ratio"] = 0.1
        param_dict["date_format"]="%Y/%m/%d %H:%M"
    elif DATASET_NAME in ("ETTH1", "ETTH2", "ETTM1", "ETTM2"):
        param_dict["data_file_path"] = os.path.join("../data/", DATASET_NAME, f"{DATASET_NAME}.csv")
        param_dict["train_ratio"] = 0.6
        param_dict["valid_ratio"] = 0.2
    else:
        raise ValueError("Unsupported dataset.")
        
    # print args
    print("-"*(20+45+5))
    for key, value in param_dict.items():
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))
    
    data_path = os.path.join(param_dict["dataset_dir"], "data.npz")
    index_path = os.path.join(param_dict["dataset_dir"], f"index_{args.history_seq_len}_{args.future_seq_len}.npz")

    param_dict["save_data"] = True
    if os.path.exists(data_path) and os.path.exists(index_path):
        reply = str(input(
            f"{os.path.join(param_dict['dataset_dir'], f'data.npz and index_{args.history_seq_len}_{args.future_seq_len}.npz')} exist. Do you want to overwrite them? (y/n) "
            )).lower().strip()
        if reply[0] != "y":
            sys.exit(0)
    elif os.path.exists(data_path) and not os.path.exists(index_path):
        print("Generating new indices...")
        param_dict["save_data"] = False
            
    generate_data(**param_dict)
    