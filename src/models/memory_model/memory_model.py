import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_model.action_sampler import ActionSampler
from src.models.memory_model.decoder import Decoder, DecoderBlock
from src.models.memory_model.encoder import Encoder, EncoderBlock
from src.models.rl.utils import State


class MemoryModel(torch.nn.Module):
    """
    A memory model that takes as input high-level embeddings from the language model
    and existing memory. Returns the normal distribution parameters for sampling new
    memory and the probability for each memory position for replacing it.
    """

    def __init__(self,
                 d_embd: int,
                 d_mem: int,
                 num_vectors: int,
                 device: torch.device,
                 memory_type: str = "conservative",
                 n_enc_block: int = 2,
                 n_dec_block: int = 3,
                 dtype: torch.dtype = torch.float32) -> None:
        """
        :param d_embd: main model embedding size
        :param d_mem: memory vector size
        :param memory_type: type of memory action sampling. Conservative memory allows only
            one memory cell to be replaced, while flexible memory allows multiple memories
            to be replaced at the same time.
        :param dtype:
        """

        if memory_type not in ["conservative", "flexible"]:
            raise ValueError("Unsupported memory type. Choose 'conservative' or 'flexible'.")

        super().__init__()
        self.d_embd = d_embd
        self.d_mem = d_mem
        self.num_vectors = num_vectors
        self.memory_type = memory_type
        self.dtype = dtype
        self.device = device
        self.encoder = Encoder(EncoderBlock(d_embd=d_embd, dtype=dtype), n_enc_block)
        self.decoder = Decoder(DecoderBlock(d_embd=d_embd, d_mem=d_mem, dtype=dtype), n_dec_block)
        self.action_sampler = ActionSampler(d_mem=d_mem, dtype=dtype, memory_type=memory_type)

    def forward(self, state: State) -> tuple[torch.tensor, torch.tensor]:
        """
        :param state:
        :return: distribution parameters for positions and new memory
        """
        embeddings, attention_mask = state.embeddings, state.attention_mask
        enc_out = self.encoder(embeddings, attention_mask)
        dec_out = self.decoder(state.memory, enc_out, attention_mask)
        action = self.action_sampler(dec_out)
        return action

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.train()
        for p in self.parameters():
            p.requires_grad = True


class SyntheticTaskModel(torch.nn.Module):
    def __init__(self, num_vectors: int, d_mem: int, memory_type: str):
        super().__init__()
        self.d_mem = d_mem
        self.num_vectors = num_vectors
        self.memory_type = memory_type
        self.fc1 = nn.Linear(d_mem, d_mem)
        self.conv1 = nn.Conv1d(d_mem, d_mem * 4, kernel_size=num_vectors, padding="same")
        self.conv2 = nn.Conv1d(d_mem * 4, d_mem, kernel_size=num_vectors, padding="same")
        self.fc2 = nn.Linear(d_mem, 1)
        self.fc3 = nn.Linear(d_mem, d_mem * 2)

    def forward(self, x: State) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x.memory))
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        out1 = self.fc2(x)

        positions = None
        if self.memory_type == "conservative":
            positions = F.softmax(out1.squeeze(dim=-1), dim=-1)  # [batch_size, num_vectors]
        elif self.memory_type == "flexible":
            positions = F.sigmoid(out1.squeeze(dim=-1))  # [batch_size, num_vectors]

        return positions.float(), self.fc3(x).float()
