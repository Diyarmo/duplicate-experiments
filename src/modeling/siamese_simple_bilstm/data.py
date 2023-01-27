from modeling.siamese_bilstm.data_utils import digit_words, digits
from data.text_normalizer import normalize_text
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        row["post1_title"] = normalize_text(row["post1_title"])
        row["post2_title"] = normalize_text(row["post2_title"])
        row["post1_desc"] = normalize_text(row["post1_desc"])
        row["post2_desc"] = normalize_text(row["post2_desc"])

        # Post 1 - Original
        post1_title = row["post1_title"]
        post1_description = row["post1_desc"]

        # Post 2 - Compare
        post2_title = row["post2_title"]
        post2_description = row["post2_desc"]

        post1_x = [post1_title, post1_description]
        post2_x = [post2_title, post2_description]
        y_duplicate = row["is_duplicate"]

        return post1_x, post2_x, y_duplicate


class DuplicateDataLoader(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_file: str,
                 batch_size,
                 text_max_length: int,
                 train_file: str = None,
                 test_file: str = None,
                 test_sample_size: int = 0
                 ):
        super().__init__()
        self.test_sample_size = test_sample_size

        self.train_file = train_file
        if train_file:
            self.train = self.read_files(self.train_file)
            print("Train Dataset has", len(self.train.data), "rows!")

        self.test_file = test_file
        if test_file:
            self.val = self.read_files(self.test_file)
            print("Val Dataset has", len(self.val.data), "rows!")

        self.tokenizer_file = tokenizer_file
        self.batch_size = batch_size
        self.text_max_length = text_max_length

        self.text_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file)
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def read_files(self, data_file):
        df = pd.read_parquet(data_file)
        df['is_duplicate'] = (df['is_duplicate']).astype(int)
        return DuplicateDataset(df)

    def encode_texts(self, texts):
        texts_encoded = self.text_tokenizer.batch_encode_plus(texts)
        texts_encoded = self.text_tokenizer.pad(
            texts_encoded, max_length=self.text_max_length, padding='longest')

        texts_masks = torch.tensor(
            texts_encoded['attention_mask'], dtype=torch.int64)
        texts_padded = torch.tensor(
            texts_encoded['input_ids'], dtype=torch.int64)
        return texts_padded, texts_masks

    def collate_batch(self, batch):
        post1_titles = []
        post1_descs = []

        post2_titles = []
        post2_descs = []

        labels = []

        for post1_x, post2_x, _y in batch:
            post1_titles.append(post1_x[0])
            post1_descs.append(post1_x[1])

            post2_titles.append(post2_x[0])
            post2_descs.append(post2_x[1])

            labels.append(_y)

        # Post1 Encodings
        post1_titles_padded, post1_titles_masks = self.encode_texts(
            post1_titles)
        post1_descs_padded, post1_descs_masks = self.encode_texts(post1_descs)
        post1_x = [post1_titles_padded, post1_descs_padded]

        # Post2 Encodings
        post2_titles_padded, post2_titles_masks = self.encode_texts(
            post2_titles)
        post2_descs_padded, post2_descs_masks = self.encode_texts(post2_descs)
        post2_x = [post2_titles_padded, post2_descs_padded]

        labels = torch.tensor(labels, dtype=torch.float)

        return post1_x, post2_x, labels

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          collate_fn=self.collate_batch,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        if self.test_sample_size:
            self.val.set_sample_data(self.test_sample_size)

        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
