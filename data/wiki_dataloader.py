import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.wiki_dataset import WikiDataset


class EpochDataloader(DataLoader):
    """Dataloader class for `WikiDataset`. By epoch we mean learning from batch articles of the data. A batch of
    articles is obtained as follows: they are tokenized, trimmed at a random place to the same length (this length
    is the minimum length of a tokenized article in a batch) and converted into a tensor."""

    def __init__(self, dataset: WikiDataset,
                 tokenizer: AutoTokenizer.from_pretrained,
                 tokenizer_parameters: dict,
                 batch_size: int = 4,
                 shuffle: bool = False,
                 **kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_parameters = tokenizer_parameters
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn, **kwargs)

    def _crop_data(self, data: [[int]]) -> [[int]]:
        lengths = [len(x) for x in data]
        max_len = min(lengths)
        start_max_token = [length - max_len for length in lengths]
        start = [random.choice(range(0, idx + 1)) for idx in start_max_token]
        return [x[st: st + max_len] for st, x in zip(start, data)]

    def _collate_fn(self, batch: [str]) -> torch.Tensor:
        tokenized_batch = [self.tokenizer(x, **self.tokenizer_parameters)["input_ids"] for x in batch]
        return torch.tensor(np.array(self._crop_data(tokenized_batch)), dtype=torch.long)

    def __iter__(self):
        self.iterator = super().__iter__()
        return self

    def __next__(self) -> torch.Tensor:
        return next(self.iterator)
