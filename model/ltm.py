import torch
import torch.nn as nn
import torch.nn.functional as F


class LTMModel(nn.Module):
    def __init__(self, n_embd: int, max_seq_len: int = 5):
        super().__init__()
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

    def get_embs(self, data):
        return torch.randn((data.shape[0], data.shape[1], 16))

    def get_reward(self, embs, memory):
        return torch.ones((embs.shape[0], 1))

    def forward(self):
        pass
