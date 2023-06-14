import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List


def load_raw_data(name: str, data_dir: str) -> Tuple[Tuple[List, List, List], int]:
    """
    Load selected raw datasets as list. Refining will be executed after loading.

    Args:
        name (str): Selected dataset name.
        data_dir (str): Direcotry of data folder.

    Raises:
        ValueError: Raise when unknown dataset name is selected.

    Returns:
        Tuple[Tuple[List, List, List], int]: Tuple of datsets, and number of features in the dataset.
    """
    if name == "Yahoo":
        feat_num = 699
        with open(f"{data_dir}/ltrc_yahoo/set1.train.txt", 'r') as f:
            train_set = f.readlines()
        with open(f"{data_dir}/ltrc_yahoo/set1.valid.txt", 'r') as f:
            valid_set = f.readlines()
        with open(f"{data_dir}/ltrc_yahoo/set1.test.txt", 'r') as f:
            test_set = f.readlines()

    elif name == "Web30K":
        feat_num = 136
        with open(f"{data_dir}/Fold1/train.txt", 'r') as f:
            train_set = f.readlines()
        with open(f"{data_dir}/Fold1/vali.txt", 'r') as f:
            valid_set = f.readlines()
        with open(f"{data_dir}/Fold1/test.txt", 'r') as f:
            test_set = f.readlines()

    elif name == "Istella":
        feat_num = 220
        with open(f"{data_dir}/full/train.txt", 'r') as f:
            train_set = f.readlines()
        valid_set = None
        with open(f"{data_dir}/full/test.txt", 'r') as f:
            test_set = f.readlines()

    else:
        raise ValueError("Unknown dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")

    return (train_set, valid_set, test_set), feat_num


def refine_raw_data(data_dir: str, datasets: Tuple[List, List, List], feat_num: int) -> None:
    """
    Refine loaded raw data. Samples are saved as npy per qid.

    Args:
        data_dir (str): Directory of data folder.
        datasets (Tuple[List, List, List]): Datasets to refine.
                                            It is consist of train, validation, test sets.
                                            In Istella, There is no validation set.
        feat_num (int): Number of features in the dataset.
    """
    for dataset, split in zip(datasets, ("train", "valid", "test")):
        if dataset is None: 
            continue
        save_dir = f"{data_dir}/{split}"
        os.makedirs(save_dir, exist_ok=True)

        samples = []
        progress_bar = tqdm(dataset)
        for sample in progress_bar:
            progress_bar.set_description(f"Refine {split} dataset")
            row = re.sub("\n| \n", "", sample).split(" ")
            new_row = [-1. for i in range(feat_num)]
            relevance = int(row[0])
            qid = row[1].split(":")[-1]

            for value in row[2:]:
                k, v = value.split(":")
                new_row[int(k)-1] += float(v) + 1.

            samples.append([relevance, qid] + new_row)

        columns = ["relevance", "qid"]+ [f"feat{str(i)}" for i in range(1, feat_num+1)]
        df = pd.DataFrame(data=samples, columns=columns)
        qid_list = df["qid"].unique()

        progress_bar = tqdm(qid_list)
        for qid in progress_bar:
            progress_bar.set_description(f"Save {split} dataset per 'qid'")
            df_qid = df[df["qid"] == qid]
            relevance = df_qid["relevance"].values
            features = df_qid.iloc[:, 2:].to_numpy()
            np.savez(f"{save_dir}/qid_{'{0:0>6}'.format(int(qid))}.npz", relevance=relevance, features=features)


if __name__ == "__main__":
    data_root = "."
    name = "Yahoo"
    data_dir = f"{data_root}/data/{name}"

    datasets, feat_num = load_raw_data(name, data_dir)
    refine_raw_data(data_dir, datasets, feat_num)
