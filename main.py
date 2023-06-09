#%%
import os
import torch
import numpy as np
import pandas as pd


#%%
# dataset_list = ["Web30K", "Yahoo", "Istella"]
data_root = "."
name = "Yahoo"

if name == "Web30K":
    data_dir = f"{data_root}/data/{name}"
elif name == "Yahoo":
    data_dir = f"{data_root}/data/{name}/ltrc_yahoo"
elif name == "Istella":
    data_dir = f"{data_root}/data/{name}"
else:
    raise ValueError("Unknow dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")

os.listdir(data_dir)

with open(f"{data_dir}/set1.train.txt", 'r') as f:
    a = f.readlines()
len(a)


a[-1]
columns = ["relevance", "qid", ]
train_set = pd.read_table(f"{data_dir}/set1.train.txt", sep="\t")

relevance_label = {
    0: "Bad",
    1: "Fair",
    2: "Good",
    3: "Excellent",
    4: "Perfect",
}

