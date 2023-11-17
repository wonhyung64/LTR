#%%
import re
import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from tqdm import tqdm
from typing import Tuple, List

from module.dataset.refining import (
    load_raw_data,
    refine_raw_data,
)
from module.utils import (
    check_data,
    check_device,
    set_seed,
)
from module.dataset.loader import (
    IRDataset,
    ir_collate_fn,
)
from module.evaluate.metric import (
    cg_fn,
    dcg_fn,
    idcg_fn,
    ndcg_fn,
)


#%%
if __name__ == "__main__":
    data_root = "."
    # name = "Yahoo"
    name = "Web30K"
    # name = "Istella"
    data_dir = f"{data_root}/data/{name}"
    batch_size = 4
    data_seed = 0

    #checking
    check_data(name)
    device = check_device()

    #check refined data
    if not all([
        os.path.exists(f"{data_dir}/train"),
        os.path.exists(f"{data_dir}/test"),
        ]):
        #refine data
        datasets, feat_num = load_raw_data(name, data_dir)
        refine_raw_data(data_dir, datasets, feat_num)

    #dataloader
    set_seed(data_seed)
    train_loader = data.DataLoader(
        dataset=IRDataset(data_dir, "train"), batch_size=batch_size, shuffle=True, collate_fn=ir_collate_fn)
    valid_loader = data.DataLoader(
        dataset=IRDataset(data_dir, "valid"), batch_size=batch_size, collate_fn=ir_collate_fn)
    test_loader = data.DataLoader(
        dataset=IRDataset(data_dir, "test"))


    for batch_x, batch_y in train_loader:break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        print(batch_x.shape, batch_y.shape)
        break

    relevance = batch_y

    print(cg_fn(relevance))
    print(dcg_fn(relevance))
    print(idcg_fn(relevance))
    print(ndcg_fn(relevance))

# %%
train_iter = iter(train_loader)
x, y = next(train_iter)
x.shape
y.shape
dataset=IRDataset(data_dir, "train")
x, y = dataset[9]
print(x.shape, y.shape)

sample_num_list = []
for x, y in dataset:
    sample_num_list.append(x.shape[0])
    sample_num_list.sort()
    len(sample_num_list)

np.mean(sample_num_list)

# %%
