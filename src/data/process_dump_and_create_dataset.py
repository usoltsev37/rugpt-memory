import json
import pickle
import re
from itertools import chain
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def process_wikilinks(wikilinks: [str], title2idx: dict, dataset: Dataset) -> [int]:
    """Check if link exists. Link is the title of relevant article, we convert it to id of article"""
    cleaned_links = []
    for link in wikilinks:
        idx = title2idx.get(link, None)
        if idx is not None:
            cleaned_links.append(int(dataset["train"][idx]["id"]))
    return cleaned_links


def get_sections(article: dict,
                 paragraphs: dict,
                 title2idx: dict,
                 sections_to_skip: [str],
                 dataset: Dataset) -> tuple[list, list]:
    text = article["text"]
    section_names, wikilinks = zip(*paragraphs.items())
    processed_wikilinks = [process_wikilinks(links_lst, title2idx, dataset) for links_lst in wikilinks]
    texts, links = [], []
    cur_text = text

    for i, name in enumerate(section_names[1:]):
        pattern = r'(\n\n\s*)?' + re.escape(name)
        match = re.search(pattern, cur_text)
        if match:
            idx = match.start()
            texts.append(cur_text[:idx])
            links.append(processed_wikilinks[i])
            cur_text = cur_text[idx:]

            # If we come across a section that needs to be skipped, then we skip all subsequent
            # sections because they are meaningless.
            if name in sections_to_skip:
                return texts, links
        else:
            # We were unable to correctly match the paragraphs and links to them, so we will save the full texts
            return [text], [list(chain.from_iterable(processed_wikilinks))]

    if cur_text:
        texts.append(cur_text)
        links.append(processed_wikilinks[-1])

    return texts, links


def get_title2idx(dataset: Dataset, reload: bool) -> dict:
    if reload:
        with open("/data/title2idx.pkl", "rb") as f:
            return pickle.load(f)

    title2idx = {}
    for idx, row in enumerate(dataset['train']):
        title2idx[row['title']] = idx

    with open("/data/title2idx.pkl", "wb") as f:
        pickle.dump(title2idx, f)

    return title2idx


def process_dataset(path_to_processed_dump: str,
                    out_path: str,
                    dataset: Dataset,
                    title2idx: dict,
                    sections_to_skip: [str]):
    """
    :param path_to_processed_dump: path to processed dump, row is an article,
    for each section there is a list with wikilinks
    :param out_path: path to save output
    :param dataset: Wikimedia data
    :param title2idx:
    :param sections_to_skip:
    """
    with open(path_to_processed_dump, "rb") as f_in, open(out_path, "w+") as f_out:
        for line in tqdm(f_in):
            article = json.loads(line)
            title = article['title']
            paragraphs = article['paragraphs']
            idx = title2idx.get(title, None)
            if idx is not None:
                dataset_row = dataset['train'][idx]
                texts, links = get_sections(dataset_row, paragraphs, title2idx, sections_to_skip, dataset)
                f_out.write(json.dumps({"id": int(dataset_row["id"]),
                                        "title": dataset_row["title"],
                                        "section_texts": texts,
                                        "section_links": links}, ensure_ascii=False))
                f_out.write('\n')
