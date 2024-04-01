import json
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import gc


def split_file(original_file_path: str, target_dir: str, max_size_bytes: int = 200 * 1024 * 1024) -> None:
    """
    Splits a large file into smaller parts with a maximum size each, saving them to a target directory.
    :param original_file_path: The path to the original large file.
    :param target_dir: The directory where the split files will be saved.
    :param max_size_bytes: The maximum size in bytes for each split file part (default is 200MB).
    """
    original_file_path = Path(original_file_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with original_file_path.open('rb') as original_file:
        part_num = 0
        part_content = []
        current_size = 0

        for line in original_file:
            part_content.append(line)
            current_size += len(line)
            if current_size >= max_size_bytes:
                part_file_path = target_dir / f"part_{part_num}.jsonl"
                with part_file_path.open('wb') as part_file:
                    part_file.writelines(part_content)
                part_content = []
                current_size = 0
                part_num += 1

        # Save any remaining content as the last part
        if part_content:
            part_file_path = target_dir / f"part_{part_num}.jsonl"
            with part_file_path.open('wb') as part_file:
                part_file.writelines(part_content)


class WikiDataset(Dataset):
    def __init__(self, data_path: str, n_context: int = 1, split='train') -> None:
        """
        Initializes the WikiDataset object for loading and processing Wikipedia article data.
        :param data_path: Path to the folder with the dataset ("dataset_sections.jsonl" divided into files)
        :param n_context: Number of article contexts to sample for each paragraph (default is 1).
        :param index_path: Path to save or load the index file.
        :param split: Train or Test
        """
        self.path = Path(data_path)
        self.n_context = n_context
        self.index_path = Path(data_path) / 'split.train_index.pkl' if split == 'train' else Path(
            data_path) / 'split.test_index.pkl'

        if not self.index_path.exists():
            raise Exception("Path to index does not exist! Try another one!")

        self.id_to_offset = self._load_index()
        self.ids = list(self.id_to_offset.keys())
        self.length = len(self.ids)

    def _load_index(self) -> dict:
        """
        Loads or creates an index mapping article IDs to their file locations and offsets.
        :return: A dictionary mapping article IDs to tuples containing the file path and byte offset.
        """
        with self.index_path.open('rb') as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return self.length

    def _load_article_by_id(self, article_id: int) -> [str]:
        """
        Loads an article's sections by its ID.
        :param article_id: The ID of the article to load.
        :return: A list of sections texts from the article.
        """
        file_path_offset = self.id_to_offset.get(article_id)
        if file_path_offset is None:
            return []

        file_path, offset = file_path_offset
        with file_path.open("rb") as f:
            f.seek(offset)
            return json.loads(f.readline())["texts"]

    def _sample_context(self, link_ids: [int]):
        """
        Samples a specified number of context articles from given link IDs.
        :param link_ids: A list of article IDs to sample from.
        :return: A string containing the concatenated text of the sampled context articles.
        """
        sampled_ids = random.sample(link_ids, min(self.n_context, len(link_ids)))
        return "\n".join([''.join(self._load_article_by_id(id_)) for id_ in sampled_ids])

    def _add_context_to_article(self, article: dict) -> str:
        """
        Adds sampled context to each section of an article.
        :param article: The article data as a dictionary.
        :return: A string containing the article text with context added to each section.
        """
        article_texts = [article["title"] + '\n']
        for (text, links) in zip(article["texts"], article["links"]):
            context = self._sample_context(links)
            if context:
                article_texts.append(context)
            article_texts.append(text)
        return "".join(article_texts)

    def __getitem__(self, item) -> str:
        """
        Retrieves an article by index, adding context to its sections.
        :param item: The index of the article in the dataset.
        :return: The text of the article with context added to each section.
        """
        article_id = self.ids[item]
        path_to_file, offset = self.id_to_offset[article_id]
        with path_to_file.open('rb') as f:
            f.seek(offset)
            article = json.loads(f.readline())
            return self._add_context_to_article(article)


def create_index(dataset_path: str, index_path: str) -> None:
    """
    Creates an index mapping article IDs to their file locations and offsets.
    :return: A dictionary mapping article IDs to tuples containing the file path and byte offset.
    """
    dataset_path = Path(dataset_path)
    index_path = Path(index_path)

    index = {}
    for part_file in dataset_path.iterdir():
        with part_file.open('rb') as f:
            offset = 0
            while line := f.readline():
                if line != b"\n":
                    article_id = json.loads(line)["document_id"]
                    index[article_id] = (part_file, offset)
                    offset += len(line)
                else:
                    offset += len(line)

    with index_path.open('wb') as f:
        pickle.dump(index, f)


def train_test_split(test_size: float, seed: int, save_path: str, index_path: str):
    np.random.seed(seed)
    with Path(index_path).open('rb') as f:
        index = pickle.load(f)
    test_size = int(test_size * len(index))
    ids = list(index.keys())
    test_ids = np.random.choice(ids, test_size, replace=False)

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    test_index = {k: v for k, v in tqdm(index.items()) if k in test_ids}
    train_index = {k: v for k, v in tqdm(index.items()) if k not in test_ids}

    with open(save_path / "test_index.pkl", "wb") as f:
        pickle.dump(test_index, f)

    with open(save_path / "train_index.pkl", "wb") as g:
        pickle.dump(train_index, g)
