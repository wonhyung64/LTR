#%%
import os
# import torch
import numpy as np
import pandas as pd

import re
from tqdm import tqdm

#%%
data_root = "."
name = "Web30K"
data_dir = f"{data_root}/data/{name}"

if name == "Yahoo":
    with open(f"{data_dir}/ltrc_yahoo/set1.train.txt", 'r') as f:
        train_set = f.readlines()
    with open(f"{data_dir}/ltrc_yahoo/set1.valid.txt", 'r') as f:
        valid_set = f.readlines()
    with open(f"{data_dir}/ltrc_yahoo/set1.test.txt", 'r') as f:
        test_set = f.readlines()

elif name == "Web30K":
    with open(f"{data_dir}/Fold1/train.txt", 'r') as f:
        train_set = f.readlines()
    with open(f"{data_dir}/Fold1/vali.txt", 'r') as f:
        valid_set = f.readlines()
    with open(f"{data_dir}/Fold1/test.txt", 'r') as f:
        test_set = f.readlines()

elif name == "Istella":
    with open(f"{data_dir}/full/train.txt", 'r') as f:
        train_set = f.readlines()
    valid_set = None
    with open(f"{data_dir}/full/test.txt", 'r') as f:
        test_set = f.readlines()

else:
    raise ValueError("Unknow dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")


#%%
samples = []
# for _, sample in tqdm(enumerate(train_set)):
for _, sample in tqdm(enumerate(train_set[:1000])):
    row = sample.split(" ")[:-1]
    new_row = [re.sub(".*:", "", row[j]) for j in range(len(row))]
    samples.append(new_row)

columns = ["relevance", "qid"] + [f"feat_{i}" for i in range(1, len(samples[0])-1)]
df = pd.DataFrame(
    data=samples,
    columns=columns,
    dtype=float,
    )
qid_1 = df[df["qid"] == 1]
relevance = qid_1["relevance"].values
features = qid_1.iloc[:, 2:].to_numpy()

np.savez("./qid_1.npz", relevance=relevance, features=features)
tmp = np.load("./qid_1.npz")
tmp["relevance"]
tmp["features"]
#%%

columns = ["relevance", "qid", ]
train_set = pd.read_table(f"{data_dir}/set1.train.txt", sep="\t")

relevance_label = {
    0: "Bad",
    1: "Fair",
    2: "Good",
    3: "Excellent",
    4: "Perfect",
}

