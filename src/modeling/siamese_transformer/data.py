import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, Dataset
import re
import Levenshtein
from datasketch import MinHash

from transformers import PreTrainedTokenizerFast

torch.multiprocessing.set_sharing_strategy('file_system')

from data.text_normalizer import normalize_text, tizi_normalizer
from modeling.siamese_bilstm.data_utils import digit_words, digits

def extract_list_of_numbers_from_text(text):
    alpha_numberic_digits = digit_words + digits
    adad_detector = r'(?:{})+((?:و|\s|\u200c)*(?:{})*)*'.format('|'.join(alpha_numberic_digits),
                                                                '|'.join(alpha_numberic_digits))
    adad_matches = re.finditer(adad_detector, text)
    matches = []
    for match in adad_matches:
        if match is not None:
            matches.append(match[0].strip())
    return matches


def levenshtein_dist(base_text: str, compare_text: str):
    if len(base_text) == len(compare_text) == 0:
        return 0.0
    else:
        return 1.0 - Levenshtein.ratio(base_text, compare_text)


def get_ngrams(words, ngram):
    return [words[i:i + ngram] for i in range(len(words) - ngram + 1)]


def text_words(text):
    words = text.split()
    ngrams = get_ngrams(words, 2)
    joined_ngrams = [' '.join(n) for n in ngrams]
    return joined_ngrams


MIN_HASH_PERMS = 25

def get_text_hash(text):
    text = normalize_text(text)
    words = text_words(text)
    m = MinHash(num_perm=MIN_HASH_PERMS)
    for w in words:
        m.update(w.encode('utf8'))

    return [minhash for minhash in m.hashvalues]




class DuplicateDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.original_data = data
                
    def set_sample_data(self, sample_size):
        if sample_size >= len(self.original_data):
            self.data = self.original_data
        else:
            self.data = self.original_data.sample(sample_size)

    def __len__(self):
        return len(self.data)

    def _texts_comparison_stats(self, row):

        post1_text = row['post1_title'] + " " + row['post1_desc']
        post2_text = row['post2_title'] + " " + row['post2_desc']

        title_lev_score = levenshtein_dist(row['post1_title'], row['post2_title'])
        desc_lev_score = levenshtein_dist(row['post1_desc'], row['post2_desc'])
        text_lev_score = levenshtein_dist(post1_text, post2_text)

        title_minhash_score = len(set(get_text_hash(row['post1_title'])) & set(get_text_hash(row['post2_title']))) / MIN_HASH_PERMS
        desc_minhash_score = len(set(get_text_hash(row['post1_desc'])) & set(get_text_hash(row['post2_desc']))) / MIN_HASH_PERMS
        text_minhash_score = len(set(get_text_hash(post2_text)) & set(get_text_hash(post1_text))) / MIN_HASH_PERMS

        post1_title_numbers = extract_list_of_numbers_from_text(row['post1_title'])
        post2_title_numbers = extract_list_of_numbers_from_text(row['post2_title'])

        post1_desc_numbers = extract_list_of_numbers_from_text(row['post1_desc'])
        post2_desc_numbers = extract_list_of_numbers_from_text(row['post2_desc'])

        title_common_numbers = len(set(post1_title_numbers) & set(post2_title_numbers))
        desc_common_numbers = len(set(post1_desc_numbers) & set(post2_desc_numbers))

        # stats = {
        #     "title_lev_score": title_lev_score,
        #     "desc_lev_score": desc_lev_score,
        #     "text_lev_score": text_lev_score,
        #     "title_minhash_score": title_minhash_score,
        #     "desc_minhash_score": desc_minhash_score,
        #     "text_minhash_score": text_minhash_score
        # }

        stats = [
            title_lev_score,
            desc_lev_score,
            text_lev_score,
            title_minhash_score,
            desc_minhash_score,
            text_minhash_score,
            title_common_numbers,
            desc_common_numbers
        ]

        return stats

    def __getitem__(self, item):
        row = self.data.iloc[item]

        row["post1_title"] = normalize_text(row["post1_title"])
        row["post2_title"] = normalize_text(row["post2_title"])
        row["post1_desc"] = normalize_text(row["post1_desc"])
        row["post2_desc"] = normalize_text(row["post2_desc"])

        # Post 1 - Original
        post1_cat_slug = row["post1_category"]
        post1_text = row["post1_title"] + " " + row["post1_desc"]
        post1_city = row["post1_city"]
        post1_neighbor = row["post1_neighbor"]

        # Post 2 - Compare
        post2_cat_slug = row["post2_category"]
        post2_text = row["post2_title"] + " " + row["post2_desc"]
        post2_city = row["post2_city"]
        post2_neighbor = row["post2_neighbor"]

        post1_x = [post1_cat_slug, post1_city, post1_neighbor, post1_text]
        post2_x = [post2_cat_slug, post2_city, post2_neighbor, post2_text]
        y_duplicate = row["is_duplicate"]

        stats = self._texts_comparison_stats(row)

        return post1_x, post2_x, stats, y_duplicate


class DuplicateDataLoader(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_file: str,
                 slug_tokenizer_file: str,
                 city_tokenizer_file: str,
                 neighbor_tokenizer_file: str,
                 batch_size,
                 text_max_length: int,
                 train_file: str = None,
                 test_file: str = None,
                 test_sample_size = 0
                 ):
        super().__init__()
        self.test_sample_size = test_sample_size
        self.train_file = train_file
        if train_file:
            self.train = self.read_files(self.train_file)

        self.test_file = test_file
        if test_file:
            self.val = self.read_files(self.test_file)

        self.tokenizer_file = tokenizer_file
        self.batch_size = batch_size
        self.text_max_length = text_max_length

        self.text_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.slug_tokenizer = LabelEncoder()
        self.slug_tokenizer.classes_ = np.load(slug_tokenizer_file, allow_pickle=True)

        self.city_tokenizer = LabelEncoder()
        self.city_tokenizer.classes_ = np.load(city_tokenizer_file, allow_pickle=True)

        self.neighbor_tokenizer = LabelEncoder()
        self.neighbor_tokenizer.classes_ = np.load(neighbor_tokenizer_file, allow_pickle=True)

    def read_files(self, data_file):
        df = pd.read_parquet(data_file)
        df['is_duplicate'] = (df['is_duplicate']).astype(int)
        return DuplicateDataset(df)

    def encode_texts(self, texts):
        texts_encoded = self.text_tokenizer.batch_encode_plus(texts)
        texts_encoded = self.text_tokenizer.pad(texts_encoded, max_length=self.text_max_length, padding='longest')

        texts_masks = torch.tensor(texts_encoded['attention_mask'], dtype=torch.int64)
        texts_padded = torch.tensor(texts_encoded['input_ids'], dtype=torch.int64)
        return texts_padded, texts_masks

    def collate_batch(self, batch):
        post1_texts = []
        post1_cat_slugs = []
        post1_city = []
        post1_neighbor = []

        post2_texts = []
        post2_cat_slugs = []
        post2_city = []
        post2_neighbor = []

        stats = []
        labels = []


        for post1_x, post2_x, _stats, _y in batch:
            post1_cat_slugs.append(post1_x[0])
            post1_city.append(post1_x[1])
            post1_neighbor.append(post1_x[2])
            post1_texts.append(post1_x[3])

            post2_cat_slugs.append(post2_x[0])
            post2_city.append(post2_x[1])
            post2_neighbor.append(post2_x[2])            
            post2_texts.append(post2_x[3])

            stats.append(_stats)
            labels.append(_y)

        # Post1 Encodings
        post1_texts_padded, post1_texts_masks = self.encode_texts(post1_texts)
        post1_slugs_encoded = torch.tensor(self.slug_tokenizer.transform(post1_cat_slugs), dtype=torch.int64)
        post1_cities_encoded = torch.tensor(self.city_tokenizer.transform(post1_city), dtype=torch.int64)
        post1_neighbors_encoded = torch.tensor(self.neighbor_tokenizer.transform(post1_neighbor), dtype=torch.int64)
        post1_x = [post1_texts_padded, post1_texts_masks, post1_slugs_encoded, post1_cities_encoded, post1_neighbors_encoded]

        # Post2 Encodings
        post2_texts_padded, post2_texts_masks = self.encode_texts(post2_texts)
        post2_slugs_encoded = torch.tensor(self.slug_tokenizer.transform(post2_cat_slugs), dtype=torch.int64)
        post2_cities_encoded = torch.tensor(self.city_tokenizer.transform(post2_city), dtype=torch.int64)
        post2_neighbors_encoded = torch.tensor(self.neighbor_tokenizer.transform(post2_neighbor), dtype=torch.int64)
        post2_x = [post2_texts_padded, post2_texts_masks, post2_slugs_encoded, post2_cities_encoded, post2_neighbors_encoded]

        stats = torch.tensor(stats)
        labels = torch.tensor(labels, dtype=torch.float)

        return post1_x, post2_x, stats, labels

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          collate_fn=self.collate_batch,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        if self.test_sample_size:
            self.val.set_sample_data(self.test_sample_size)
            
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
