#%%
import os
import torch
import numpy as np
import pandas as pd


#%%
data_root = "."
name = "Web30K"

if name == "Yahoo":
    data_dir = f"{data_root}/data/{name}/ltrc_yahoo"
    with open(f"{data_dir}/set1.train.txt", 'r') as f:
        train_set = f.readlines()
    with open(f"{data_dir}/set1.valid.txt", 'r') as f:
        valid_set = f.readlines()
    with open(f"{data_dir}/set1.test.txt", 'r') as f:
        test_set = f.readlines()
elif name == "Web30K":
    data_dir = f"{data_root}/data/{name}/Fold1"
    with open(f"{data_dir}/train.txt", 'r') as f:
        train_set = f.readlines()
    with open(f"{data_dir}/vali.txt", 'r') as f:
        valid_set = f.readlines()
    with open(f"{data_dir}/test.txt", 'r') as f:
        test_set = f.readlines()
elif name == "Istella":
    data_dir = f"{data_root}/data/{name}/full"
    with open(f"{data_dir}/train.txt", 'r') as f:
        train_set = f.readlines()
    valid_set = None
    with open(f"{data_dir}/test.txt", 'r') as f:
        test_set = f.readlines()
else:
    raise ValueError("Unknow dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")
train_set[:86]
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

