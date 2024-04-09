import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.wiki_dataset import WikiDataset


class EpochDataloader(DataLoader):
    """Dataloader class for `WikiDataset`. By epoch we mean learning from batch articles of the data. A batch of
    articles is obtained as follows: they are tokenized, trimmed at a random place to the same length (this length
    is the minimum length of a tokenized article in a batch) and converted into a tensor."""

    def __init__(self,
                 dataset: WikiDataset,
                 tokenizer: AutoTokenizer.from_pretrained,
                 model_max_length: int = 2048,
                 max_sequence_len_in_batch: int = 2048 * 10,
                 batch_size: int = 5,
                 shuffle: bool = False,
                 **kwargs):
        self.iterator = None
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.max_sequence_len_in_batch = max_sequence_len_in_batch
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn, **kwargs)

    def _collate_fn(self, batch: [str]) -> dict:
        tokenized_batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.max_sequence_len_in_batch,
                                         pad_to_multiple_of=self.model_max_length)

        # Reshape to [batch_size, episode_len, len_seq_in_episode]
        tokenized_batch['input_ids'] = tokenized_batch['input_ids'].view(len(batch), -1, self.model_max_length)
        tokenized_batch['attention_mask'] = tokenized_batch['attention_mask'].view(len(batch), -1,
                                                                                   self.model_max_length)
        return tokenized_batch

    def __iter__(self):
        self.iterator = super().__iter__()
        return self

    def __next__(self) -> torch.Tensor:
        return next(self.iterator)
