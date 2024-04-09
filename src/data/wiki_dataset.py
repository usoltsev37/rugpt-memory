import json
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, data_path: str, n_context: int = 1, split: str = 'train', shuffle_context: bool = True) -> None:
        """
        Initializes the dataset object for loading and processing article data.
        Automatically handles train/test split if not already done.
        :param data_path: Path to the folder with the dataset ("dataset_sections.jsonl" divided into files)
        :param n_context: Number of article contexts to sample for each paragraph (default is 1).
        :param split: Train or Test
        """
        self.data_path = Path(data_path)
        self.shuffle_context = shuffle_context
        self.n_context = n_context
        self.split = split
        self.index_path = Path(data_path) / 'split' / f"{split}_index.pkl"

        if not self.index_path.exists():
            print("Index or split not found. Creating now...")
            self.prepare_dataset()

        self.id_to_offset = self._load_index()
        self.ids = list(self.id_to_offset.keys())
        self.length = len(self.ids)

    def prepare_dataset(self):
        """
        Prepares the dataset by creating an index and splitting into train/test.
        """
        index = self.create_index()
        self.train_test_split(index)

    def create_index(self) -> dict:
        """
        Creates an index mapping article IDs to their file locations and offsets.
        :return: A dictionary mapping article IDs to tuples containing the file path and byte offset.
        """

        index = {}
        for part_file in self.data_path.glob('*.jsonl'):
            offset = 0
            with part_file.open('rb') as f:
                for line in f:
                    article = json.loads(line)
                    index[article['document_id']] = (part_file, offset)
                    offset += len(line)

        return index

    def train_test_split(self, index: dict, train_size: float = 0.8) -> None:
        """
        Splits the index into train and test sets and saves them.
        """
        ids = list(index.keys())
        random.shuffle(ids)
        split_idx = int(len(ids) * train_size)
        train_ids, test_ids = ids[:split_idx], ids[split_idx:]

        # Get validation set
        random.shuffle(test_ids)
        split_idx = int(len(test_ids) * 0.5)
        val_ids, test_ids = ids[:split_idx], ids[split_idx:]

        train_index = {id_: index[id_] for id_ in train_ids}
        val_index = {id_: index[id_] for id_ in val_ids}
        test_index = {id_: index[id_] for id_ in test_ids}

        train_index_path = self.data_path / 'split' / "train_index.pkl"
        val_index_path = self.data_path / 'split' / "val_index.pkl"
        test_index_path = self.data_path / 'split' / "test_index.pkl"

        train_index_path.parent.mkdir(parents=True, exist_ok=True)
        with train_index_path.open('wb') as f:
            pickle.dump(train_index, f)
        with test_index_path.open('wb') as g:
            pickle.dump(test_index, g)
        with val_index_path.open('wb') as e:
            pickle.dump(val_index, e)

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
        contexts = list(filter(lambda x: x != '', [self._sample_context(links) for links in article["links"]]))

        if self.shuffle_context:
            random.shuffle(contexts)

        result = [article["title"], *contexts, ''.join(article["texts"])]
        return "\n".join(result)

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
