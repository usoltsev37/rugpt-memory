import itertools
import logging
import os
import pickle
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset
from src.models.load_ltm_model import load_ltm_model
from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.models.memory_model.memory import MemoryModule
from src.models.memory_model.memory_model import MemoryModel
from src.models.rl.agent import Agent
from src.models.rl.envs import LTMEnvironment
from src.models.rl.reinforce import REINFORCE
from src.models.rl.train import train_rl
from src.models.rl.utils import State
from src.utils.logger_singleton import ColourFormatter, logger
from src.utils.train_config import *
from src.utils.train_utils import create_dir_with_name, create_name, crop_batch, init_arguments
from transformers import GPT2LMHeadModel, GPT2Config


configuration = GPT2Config()

class LTMModel(GPT2LMHeadModel):
    def __init__(self, model,  configuration):
        super().__init__(configuration)
        self.model = model
        
    def forward(self, input_ids,
                attention_mask,
                past_key_values=None,
                use_cache=False,
                position_ids=None,
                token_type_ids=None,
                head_mask = None,
                inputs_embeds = None,
                encoder_hidden_states = None,
                encoder_attention_mask = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None,
                return_dict = None,):
        return self.model(input_ids, attention_mask)
        

args = init_arguments()
args = load_config(args.config)
set_seed(args.seed)
    
checkpoint = '/home/akarpov/jbelova/rugpt-memory/checkpoint-370'

ltm_model, tokenizer = load_ltm_model(args)
state_dict = torch.load(checkpoint + "/ltm.pt")["model_parameters"]
ltm_model.load_state_dict(state_dict)

memory_model = MemoryModel(**asdict(args.memory_model_params), dtype=ltm_model.dtype)
state_dict = torch.load(checkpoint + "/memory_model.pt")["model_parameters"]
memory_model.load_state_dict(state_dict)

ltm_model.freeze()
memory_model.freeze()
memory_module = MemoryModule(
    memory_model.d_mem,
    memory_model.num_vectors,
    memory_model.dtype,
    memory_model.memory_type,
)

agent = Agent(memory_model)
ltm_model = LTMModel(ltm_model, configuration)
ltm_model.model.memory = memory_module
ltm_model.generation_config.pad_token_id = tokenizer.pad_token_id

dataset_path = (Path(args.content_dir) / "data" / "dataset").resolve()
test_dataset = WikiDataset(data_path=str(dataset_path), split="test")
dataloader = EpochDataloader(
    test_dataset,
    tokenizer,
    step_length=args.ltm_params.step_length,
    batch_size=1,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)
it = iter(dataloader)
for _ in range(40):
    data = it.__next__()
s = data['input_ids'].shape[1]
    
bs, num_steps, _ = data['input_ids'].shape
memory_module.reset(bs)
for step in range(num_steps - 2):
    input_ids, attention_mask = (
        data["input_ids"][:, step, :].contiguous(),
        data["attention_mask"][:, step, :].contiguous(),
    )

    embeddings = ltm_model.model.get_embeddings(input_ids, attention_mask)

    # Prepare action for agent
    state = State(
        memory=ltm_model.model.memory.memory,
        attention_mask=attention_mask,
        embeddings=embeddings,
    )

    # Get new memory vectors and update memory
    with torch.no_grad():
        action, _, _ = agent.act(state)

    # Update memory
    ltm_model.model.memory.update(action)

# print("FULL")
# for _ in range(num_steps-2):
#     print(tokenizer.decode(data['input_ids'][:, _, :][0]))
#     print()
# print("<"*100)

print("GOAL")
print(tokenizer.decode(data['input_ids'][:, -2, :][0]))
print("<"*100)

input_ids, attention_mask = (
    data["input_ids"][:, -2, :50].contiguous(),
    data["attention_mask"][:,   -2, :50].contiguous(),
)
# Generation!
output = ltm_model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_length=100, top_p=0.9, num_beams=1)
# tokenizer.decode(input_ids[0])
print(tokenizer.decode(output[0][:50]))
print("-" * 50)
print(tokenizer.decode(output[0][50:]))
