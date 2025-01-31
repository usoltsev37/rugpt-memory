import pickle
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from wiki_dataloader import EpochDataloader
from wiki_dataset import WikiDataset

split = "test"
batch_size = 1

dataset_path = Path(".") / "data" / "dataset"
dataset_path.resolve()
dataset_path = str(dataset_path)
train_dataset = WikiDataset(data_path=dataset_path, split=split)
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

dataloader = EpochDataloader(
    train_dataset,
    tokenizer,
    step_length=256,
    batch_size=batch_size
)


episode_len = []
for i, data in enumerate(tqdm(dataloader)):
    episode_len.append(data["input_ids"].shape[1])

path = dataset_path + "_" + split + "_" + str(batch_size) + ".pkl"


with open(path, "wb") as f:
    pickle.dump(episode_len, f)
