import os
import torch
import numpy as np
import torch.utils.data as data


class IRDataset(data.Dataset):
    """
    Information Retrieval Datasets.
    Yahoo, Istella, MSLR-Web30K are supproted.
    """
    def __init__(self, data_dir, split):
        super(IRDataset, self).__init__()
        
        self.file_dir = f"{data_dir}/{split}"
        self.file_list = [
            file for file in os.listdir(self.file_dir) if file.__contains__("npz")
            ]
        self.file_list.sort()

    def __getitem__(self, index):
        sample = np.load(f"{self.file_dir}/{self.file_list[index]}")
        self.x_data = torch.from_numpy(sample["features"]).float()
        self.y_data = torch.from_numpy(sample["relevance"]).int()
        return self.x_data, self.y_data

    def __len__(self):
        return len(self.file_list)


def ir_collate_fn(batch):
    """
    Padding function for variable batch length.
    """
    batch_x, batch_y = [], []
    for x, y in batch:
        batch_x.append(x)
        batch_y.append(y)
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=-1.)
    batch_y = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True, padding_value=-1)

    return batch_x, batch_y
