import json
import pickle
import random

from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, path_to_data: str, n_context: int = 1, reload_index=True) -> None:
        """
        :param path_to_data: path to a pre-processed Wikipedia dump with articles consisting
        of separate paragraphs and links in them.
        :param n_context: the number of article contexts that need to be sampled for each paragraph.
        If there were no links in a paragraph, then there would be no context for it.
        """
        self.path = path_to_data
        self.reload_index = reload_index
        self.n_context = n_context
        self.id2offset = self._get_id2offset()
        self.ids = self._get_ids()
        self.length = len(self.ids)

    def _get_id2offset(self) -> dict:
        """
        Creates a dictionary in which the keys will be the `id` fields of the articles, and the values will be the byte
        positions (offsets) for the reading pointer in the `self.path` file.
        """
        if self.reload_index:
            with open("/data/id2offset.pkl", "rb") as f:
                return pickle.load(f)

        index = dict()
        with open(self.path, "rb") as f:
            offset = 0
            while line := f.readline():
                id_ = json.loads(line)["id"]
                index[id_] = offset
                offset += len(line)

        with open("/data/id2offset.pkl", "wb") as f:
            pickle.dump(index, f)

        return index

    def _get_ids(self) -> [int]:
        """Returns a list of article indexes in the order of their appearance in the data."""
        return list(self.id2offset.keys())

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _concat_sections(article: dict) -> str:
        """Creates the full text of an article from its sections by concatenation."""
        return ''.join(article["section_texts"])

    def _load_article_by_id(self, id_: int) -> dict:
        """Finds an article by its `id` in the `self.path` file."""
        with open(self.path, "rb") as f:
            f.seek(self.id2offset[id_])
            return json.loads(f.readline())

    def _sample_context(self, links: [int]):
        """Samples `self.n_context` contexts (full articles from the data) from the `links` list."""
        sampled_ids = random.sample(links, min(self.n_context, len(links)))
        if sampled_ids:
            sampled_texts = [self._concat_sections(self._load_article_by_id(id_)) for id_ in sampled_ids]
            return "\n".join(sampled_texts)
        return ""

    def _add_context_to_article(self, article: dict) -> str:
        """For the i-th section of the article, samples `self.n_context` contexts, connects them and inserts them
        before the section. All sections with contexts are concatenated at the end."""

        # Start data item with the title of the article
        texts_to_join = [article["title"]]
        for (text, links) in zip(article["section_texts"], article["section_links"]):
            context = self._sample_context(links)
            if context:
                texts_to_join.append(context)
            texts_to_join.append(text)
        return "\n".join(texts_to_join)

    def __getitem__(self, item) -> str:
        with open(self.path, "rb") as f:
            id_ = self.ids[item]
            f.seek(self.id2offset[id_])
            article = json.loads(f.readline())
            full_article = self._add_context_to_article(article)
            return full_article
