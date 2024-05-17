import json
import pickle
import random
import os
from pathlib import Path

from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

class NextFile():
    """
    Synchronous generation of next available file name.
    """

    # filesPerDir = 100

    def __init__(self, path_name, extension = ".jsonl"):
        self.path_name = path_name
        self.extension = extension
        self.dir_index = -1
        self.file_index = -1

    def next(self):
        self.file_index = self.file_index + 1
        # self.file_index = (self.file_index + 1) % NextFile.filesPerDir
        if self.file_index == 0:
            self.dir_index += 1
        dirname = self._dirname()
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return self._filepath()

    def _dirname(self):
        return self.path_name
        # char1 = self.dir_index % 26
        # char2 = int(self.dir_index / 26) % 26
        # return os.path.join(self.path_name, '%c%c' % (ord('A') + char2, ord('A') + char1))

    def _filepath(self):
        name = '%s/part_%02d' % (self._dirname(), self.file_index)
        return name + self.extension


class OutputSplitter():
    """
    File-like object, that splits output to multiple files of a given max size.
    """

    def __init__(self, nextFile, max_file_size=0, compress=False):
        """
        :param nextFile: a NextFile object from which to obtain filenames
            to use.
        :param max_file_size: the maximum size of each file.
        :para compress: whether to write data with bzip compression.
        """
        self.nextFile = nextFile
        self.compress = compress
        self.max_file_size = max_file_size
        self.file = self.open(self.nextFile.next())

    def reserve(self, size):
        if self.file.tell() + size > self.max_file_size:
            self.close()
            self.file = self.open(self.nextFile.next())

    def write(self, data):
        self.reserve(len(data))
        self.file.write(data)

    def close(self):
        self.file.close()

    def open(self, filename):
        return open(filename, 'w')
        
class WikiDataset(Dataset):
    def __init__(self, data_path: str, n_context: int = 1, split: str = "train", shuffle_context: bool = True) -> None:
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
        self.index_path = Path(data_path) / "split" / f"{split}_index.pkl"

        if not self.index_path.exists():
            print("Index or split not found. Creating now...")
            self.prepare_dataset()

        self.id_to_offset, self.ids = self._load_index()
        # self.save_dataset()
        
        self.length = len(self.ids)
        self.current_file = None
        self.current_file_index = 0
        self.file_paths = list((self.data_path / "dataset_with_sampled_context" / "train").rglob('*'))
        self.load_next_file()
        
    def load_next_file(self):
        if self.current_file:
            self.current_file.close()
        if self.current_file_index < len(self.file_paths):
            self.current_file = open(self.file_paths[self.current_file_index], 'r')
            self.current_file_index += 1
        else:
            self.current_file = None

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
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
        index = {}
        for part_file in tqdm(self.data_path.glob("*.jsonl"), total=48):
            offset = 0
            with part_file.open("rb") as f:
                for line in f:
                    article = json.loads(line)
    
                    # Check if there are no links for the article
                    usable = False
                    
                    for link_list in article["links"]:
                        if link_list:
                            usable = True
                            break

                    # If the article is very short (<300 tokens), we are not going to use it for the training (as context it's ok)
                    len_article = tokenizer("".join(article["texts"]), return_length=True)["length"][0]
                    if len_article < 1024:
                        usable = False

                    index[article["document_id"]] = (part_file.name, offset, usable)
                    offset += len(line)

        return index
    

    def save_dataset(self, ):
        print("Saving dataset...")
        dataset_path = Path(self.data_path) / "dataset_with_sampled_context" / self.split
        dataset_path.mkdir(exist_ok=True, parents=True)
        
        nextFile = NextFile(dataset_path)
        out = OutputSplitter(nextFile, 1024 * 1024 * 200)
        
        for article_id in tqdm(self.ids, total=len(self.ids)):
            path_to_file, offset, _ = self.id_to_offset[article_id]
            
            full_path_to_file = Path(self.data_path) / path_to_file
            
            with full_path_to_file.open("rb") as f:
                f.seek(offset)
                article = json.loads(f.readline())
                article_with_context = self._add_context_to_article(article)
                sample = {"title": article["title"], "text": article_with_context}
                json_article = json.dumps(sample, ensure_ascii=False)
                out.write(json_article + '\n')
        
        out.close()
        print(f"Dataset saved to {dataset_path}!")


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

        train_index = {
            "index": {id_: index[id_] for id_ in train_ids},
            "usable_ids": [id_ for id_ in train_ids if index[id_][2]],
        }
        val_index = {
            "index": {id_: index[id_] for id_ in val_ids},
            "usable_ids": [id_ for id_ in val_ids if index[id_][2]],
        }
        test_index = {
            "index": {id_: index[id_] for id_ in test_ids},
            "usable_ids": [id_ for id_ in test_ids if index[id_][2]],
        }

        train_index_path = self.data_path / "split" / "train_index.pkl"
        val_index_path = self.data_path / "split" / "val_index.pkl"
        test_index_path = self.data_path / "split" / "test_index.pkl"

        train_index_path.parent.mkdir(parents=True, exist_ok=True)
        with train_index_path.open("wb") as f:
            pickle.dump(train_index, f)
        with test_index_path.open("wb") as g:
            pickle.dump(test_index, g)
        with val_index_path.open("wb") as e:
            pickle.dump(val_index, e)

    def _load_index(self) -> tuple[dict, list[int]]:
        """
        Loads or creates an index mapping article IDs to their file locations and offsets.
        :return: A dictionary mapping article IDs to tuples containing the file path and byte offset.
        """
        with self.index_path.open("rb") as f:
            d = pickle.load(f)
            return d["index"], d["usable_ids"]

    def __len__(self) -> int:
        return self.length

    def _load_article_by_id(self, article_id: int) -> list[str]:
        """
        Loads an article's sections by its ID.
        :param article_id: The ID of the article to load.
        :return: A list of sections texts from the article.
        """
        file_path_offset = self.id_to_offset.get(article_id)
        if file_path_offset is None:
            return []

        file_path, offset, _ = file_path_offset
        full_file_path = Path(self.data_path) / file_path
        with full_file_path.open("rb") as f:
            f.seek(offset)
            return json.loads(f.readline())["texts"]

    def _sample_context(self, link_ids: list[int]):
        """
        Samples a specified number of context articles from given link IDs.
        :param link_ids: A list of article IDs to sample from.
        :return: A string containing the concatenated text of the sampled context articles.
        """
        sampled_ids = random.sample(link_ids, min(self.n_context, len(link_ids)))
        return "\n".join(["".join(self._load_article_by_id(id_)) for id_ in sampled_ids])

    def _add_context_to_article(self, article: dict) -> str:
        """
        Adds sampled context to each section of an article.
        :param article: The article data as a dictionary.
        :return: A string containing the article text with context added to each section.
        """
        contexts = list(filter(lambda x: x != "", [self._sample_context(links) for links in article["links"]]))

        if self.shuffle_context:
            random.shuffle(contexts)

        result = [article["title"], *contexts, "".join(article["texts"])]
        return "\n".join(result)
    
    def __getitem__(self, idx):
        if self.current_file is None:
            raise StopIteration("No more data available")

        line = self.current_file.readline()

        if not line:
            self.load_next_file()
            return self.__getitem__(idx)  # Recursively call to get next item from the next file
        
        article = json.loads(line)
        return article["text"]
    
    def __del__(self):
        if self.current_file:
            self.current_file.close()


    # def __getitem__(self, item) -> str:
    #     """
    #     Retrieves an article by index, adding context to its sections.
    #     :param item: The index of the article in the dataset.
    #     :return: The text of the article with context added to each section.
    #     """
    #     article_id = self.ids[item]
    #     path_to_file, offset, _ = self.id_to_offset[article_id]
    #     full_path_to_file = Path(self.data_path) / path_to_file
    #     with full_path_to_file.open("rb") as f:
    #         f.seek(offset)
    #         article = json.loads(f.readline())
    #         return self._add_context_to_article(article)

