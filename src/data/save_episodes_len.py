import pickle
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.data.wiki_dataloader import EpochDataloader
from src.data.wiki_dataset import WikiDataset

dataset_path = Path(".") / "data" / "dataset"
dataset_path.resolve()
dataset_path = str(dataset_path)
train_dataset = WikiDataset(data_path=dataset_path, split="train")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

dataloader = EpochDataloader(
    train_dataset,
    tokenizer,
    model_max_length=512,
    max_sequence_len_in_batch=512 * 100,
    batch_size=4,
    shuffle=True,
    cut_by_shortest_article=True,
)


episode_len = []
for i, data in enumerate(tqdm(dataloader)):
    episode_len.append(data["input_ids"].shape[1])
    if i % 200 == 0:
        with open(dataset_path + "/episode_len.pkl", "wb") as f:
            pickle.dump(len, f)
    if i == 50000:
        break

with open(dataset_path + "lens.pkl", "wb") as f:
    pickle.dump(len, f)
