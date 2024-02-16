import torch
from model.decoder import Decoder, DecoderBlock
from model.encoder import Encoder, EncoderBlock
from model.action_sampler import ActionSampler


class MemoryModel(torch.nn.Module):
    """
    A memory model that takes as input high-level embeddings from the language model
    and existing memory. Returns the normal distribution parameters for sampling new
    memory and the probability for each memory position for replacing it.
    """

    def __init__(self,
                 n_embd: int,
                 n_mem: int,
                 n_head: int,
                 n_enc_blocks: int = 2,
                 n_dec_blocks: int = 3,
                 dropout: float = 0.1,
                 mem_type: str = "conservative",
                 **kwargs
                 ) -> None:
        """
        :param n_embd: main model embedding size
        :param n_mem: memory embedding size
        :param n_head: number of heads in the attention mechanism
        :param n_enc_blocks: number of blocks in encoder
        :param n_dec_blocks: number of blocks in decoder
        :param mem_type: type of memory action sampling. Conservative memory allows only
            one memory cell to be replaced, while flexible memory allows multiple memories
            to be replaced at the same time.
        :param dropout: dropout value
        :return:
        """

        if mem_type not in ["conservative", "flexible"]:
            raise ValueError("Unsupported memory type. Choose 'conservative' or 'flexible'.")

        super().__init__()
        self.mem_type = mem_type
        self.n_mem = n_mem
        self.encoder = Encoder(EncoderBlock(n_embd, n_head, dropout, **kwargs), n_enc_blocks)
        self.decoder = Decoder(DecoderBlock(n_embd, n_mem, n_head, dropout, **kwargs), n_dec_blocks)
        self.action_sampler = ActionSampler(n_mem, mem_type, **kwargs)

    def forward(self, memory: torch.tensor, embeddings: torch.tensor, attn_mask: torch.tensor = None) -> tuple[
        torch.tensor, torch.tensor]:
        """
        :param memory: existing memory
        :param embeddings: high-level embeddings from the main model
        :param attn_mask: attention mask (paddings) from tokenizer
        :return: distribution parameters for positions and new memory
        """
        enc_out = self.encoder(embeddings, attn_mask)
        dec_out = self.decoder(memory, enc_out)
        action = self.action_sampler(dec_out)
        return action
