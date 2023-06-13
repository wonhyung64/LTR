#%%
import re
import os
#import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


#%%
#load raw data
data_root = "."
# name = "Yahoo"
name = "Web30K"
# name = "Istella"
data_dir = f"{data_root}/data/{name}"

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
    raise ValueError("Unknow dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")


#%%
# refine data
for dataset, split in [(train_set, "train"), (valid_set, "valid"), (test_set, "test")]:
    if dataset is None: 
        continue
    save_dir = f"{data_dir}/{split}"
    os.makedirs(save_dir, exist_ok=True)

    samples = []
    for _, sample in tqdm(enumerate(dataset)):
        row = re.sub("\n| \n", "", sample).split(" ")
        new_row = [-1. for i in range(feat_num)]
        relevance = int(row[0])
        qid = row[1].split(":")[-1]

        for __, value in enumerate(row[2:]):
            k, v = value.split(":")
            new_row[int(k)-1] += float(v) + 1.

        samples.append([relevance, qid] + new_row)

    columns = ["relevance", "qid"]+ [f"feat{str(i)}" for i in range(1, feat_num+1)]
    df = pd.DataFrame(data=samples, columns=columns)
    qid_list = df["qid"].unique()

    for qid in tqdm(qid_list):
        df_qid = df[df["qid"] == qid]
        relevance = df_qid["relevance"].values
        features = df_qid.iloc[:, 2:].to_numpy()
        np.savez(f"{save_dir}/qid_{'{0:0>6}'.format(int(qid))}.npz", relevance=relevance, features=features)


#%%
file_list = os.listdir(f"{data_dir}/train")
sample = np.load(f"{data_dir}/train/{file_list[0]}")
print(sample["features"].shape)
print(sample["relevance"].shape)
