from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MovieDataset(Dataset):
    def __init__(self, plots: pd.Series):
        self.plots = plots

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, idx):
        return self.plots[idx]


def get_dataloader(csv_file, batch_size, max_data_size=None):
    plots = pd.read_csv(csv_file)["Plot"]
    if max_data_size:
        plots = plots[:max_data_size]
    dataset = MovieDataset(plots)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def tokenize_sentences(tokenizer, sentences: List[str], context_length):
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=context_length,
    )
    return tokens
