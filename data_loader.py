#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils import data


class DataFolder(data.Dataset):
    """Load Data for Iterator. """
    def __init__(self, data_path):
        """Initializes image paths and preprocessing module."""
        self.data = np.load(data_path)

    def __getitem__(self, index):
        """Reads an Data and Neg Sample from a file and returns."""
        x,len_x,y,len_y,l = self.data[index]

        return x,len_x,y,len_y,l

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.data)


def get_loader(root, data_path, batch_size, shuffle=True, num_workers=2):
    """Builds and returns Dataloader."""

    dataset = DataFolder(root+data_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader
