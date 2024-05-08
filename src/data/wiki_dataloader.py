import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from src.data.wiki_dataset import WikiDataset


class EpochDataloader(DataLoader):
    """Dataloader class for WikiDataset. By epoch we mean learning from batch articles of the data. A batch of
    articles is obtained as follows: they are tokenized, trimmed at a random place to the same length (this length
    is the minimum length of a tokenized article in a batch) and converted into a tensor.
    """

    def __init__(
        self,
        dataset: WikiDataset,
        tokenizer: AutoTokenizer.from_pretrained,
        step_length: int = 512,
        batch_size: int = 4,
        shuffle: bool = False,
        **kwargs
    ):
        self.iterator = None
        self.tokenizer = tokenizer
        self.step_length = step_length
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn, **kwargs)

    def _collate_fn(self, batch: list[str]) -> dict:
        tokenized_batch = self.tokenizer(batch, return_tensors="pt", padding=True)
        shortest_article_len = tokenized_batch["attention_mask"].sum(dim=-1).min()

        tokenized_batch["input_ids"] = tokenized_batch["input_ids"][:, :shortest_article_len]
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"][:, :shortest_article_len]

        add_tokens_num = (self.step_length - shortest_article_len) % self.step_length

        if add_tokens_num:
            if self.step_length - add_tokens_num == 1:
                tokenized_batch["input_ids"] = tokenized_batch["input_ids"][:, :-1]
                tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"][:, :-1]
            else:
                tokenized_batch["input_ids"] = F.pad(
                    tokenized_batch["input_ids"], (0, add_tokens_num), "constant", self.tokenizer.pad_token_id
                ).long()
                tokenized_batch["attention_mask"] = F.pad(
                    tokenized_batch["attention_mask"], (0, add_tokens_num), "constant", self.tokenizer.pad_token_id
                ).long()

        # Reshape to [batch_size, episode_len, len_seq_in_episode]
        tokenized_batch["input_ids"] = tokenized_batch["input_ids"].view(len(batch), -1, self.step_length)
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].view(len(batch), -1, self.step_length)
        return tokenized_batch

    def __iter__(self):
        self.iterator = super().__iter__()
        return self

    def __next__(self) -> torch.Tensor:
        return next(self.iterator)
