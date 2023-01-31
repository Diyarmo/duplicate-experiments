from data.text_normalizer import normalize_text
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

from transformers import PreTrainedTokenizerFast

torch.multiprocessing.set_sharing_strategy('file_system')


class DuplicateDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.original_data = data

    def set_sample_data(self, sample_size):
        if sample_size >= len(self.original_data):
            self.data = self.original_data
        else:
            self.data = self.original_data.sample(sample_size)

    def resample_by_label(self):
        min_sample_count = self.original_data['is_duplicate'].sum()

        sampled_inddices = self.original_data.apply(
            lambda x: np.random.choice(x.index, 3 * min_sample_count)
            if len(x) > min_sample_count
            else x.index)

        sampled_inddices = np.concatenate(sampled_inddices)
        self.data = self.original_data.loc[sampled_inddices].sample(
            len(sampled_inddices)).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        post1_title = normalize_text(row["post1_title"])
        post1_desc = normalize_text(row["post1_desc"])
        post2_title = normalize_text(row["post2_title"])
        post2_desc = normalize_text(row["post2_desc"])

        # Post 1 - Original
        post1_text = f'عنوان آگهی: {post1_title} | توضیحات آگهی: {post1_desc}'
        post1_cat_slug = row["post1_category"]
        post1_city = row["post1_city"]

        # Post 2 - Compare
        post2_text = f'عنوان آگهی: {post2_title} | توضیحات آگهی: {post2_desc}'
        post2_cat_slug = row["post2_category"]
        post2_city = row["post2_city"]

        post1_x = [post1_text, post1_cat_slug, post1_city]
        post2_x = [post2_text, post2_cat_slug, post2_city]

        y_duplicate = row["is_duplicate"]

        return post1_x, post2_x, y_duplicate


class DuplicateDataLoader(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_file: str,
                 slug_tokenizer_file: str,
                 city_tokenizer_file: str,
                 batch_size,
                 text_max_length: int,
                 train_file: str = None,
                 test_file: str = None,
                 test_sample_size: int = 0,
                 resample_by_label: bool = False
                 ):
        super().__init__()
        self.test_sample_size = test_sample_size
        self.resample_by_label = resample_by_label
        self.batch_size = batch_size
        self.text_max_length = text_max_length

        self.text_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file)
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.slug_tokenizer = LabelEncoder()
        self.slug_tokenizer.classes_ = np.load(
            slug_tokenizer_file, allow_pickle=True)

        self.city_tokenizer = LabelEncoder()
        self.city_tokenizer.classes_ = np.load(
            city_tokenizer_file, allow_pickle=True)

        self.train_file = train_file
        if train_file:
            self.train = self.read_files(self.train_file)
            print("Train Dataset has", len(self.train.data), "rows!")

        self.test_file = test_file
        if test_file:
            self.val = self.read_files(self.test_file)
            print("Val Dataset has", len(self.val.data), "rows!")

    def read_files(self, data_file):
        df = pd.read_parquet(data_file)
        df['is_duplicate'] = (df['is_duplicate']).astype(int)
        return DuplicateDataset(df)

    def encode_texts(self, texts):
        tokenized_text = self.text_tokenizer(
            texts, max_length=self.text_max_length, truncation=True, padding="longest", return_tensors="pt")
        return tokenized_text

    def collate_batch(self, batch):
        post1_texts = []
        post1_slugs = []
        post1_cities = []

        post2_texts = []
        post2_slugs = []
        post2_cities = []

        labels = []

        for post1_x, post2_x, _y in batch:
            post1_texts.append(post1_x[0])
            post1_slugs.append(post1_x[1])
            post1_cities.append(post1_x[2])

            post2_texts.append(post1_x[0])
            post2_slugs.append(post1_x[1])
            post2_cities.append(post1_x[2])

            labels.append(_y)

        post1_tokenized_texts = self.encode_texts(post1_texts)
        
        post1_slugs_encoded = torch.tensor(
            self.slug_tokenizer.transform(post1_slugs), dtype=torch.int64)
        post1_cities_encoded = torch.tensor(
            self.city_tokenizer.transform(post1_cities), dtype=torch.int64)

        post1_x = [post1_tokenized_texts["input_ids"],
                   post1_tokenized_texts["attention_mask"], 
                   post1_slugs_encoded, 
                   post1_cities_encoded]

        # Post2 Encodings
        post2_tokenized_texts = self.encode_texts(post2_texts)

        post2_slugs_encoded = torch.tensor(
            self.slug_tokenizer.transform(post2_slugs), dtype=torch.int64)
        post2_cities_encoded = torch.tensor(
            self.city_tokenizer.transform(post2_cities), dtype=torch.int64)

        post2_x = [post2_tokenized_texts["input_ids"],
                   post2_tokenized_texts["attention_mask"], 
                   post2_slugs_encoded, 
                   post2_cities_encoded]

        labels = torch.tensor(labels, dtype=torch.float)

        return post1_x, post2_x, labels

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          collate_fn=self.collate_batch,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        if self.resample_by_label:
            self.val.resample_by_label()
        if self.test_sample_size:
            self.val.set_sample_data(self.test_sample_size)
        
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
