import sys

sys.path.extend(['.'])

from typing import List

from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import yaml
import pandas as pd

from text_normalizer import normalize_text

def train_and_save_word_piece_tokenizer(vocab_size: int, text_data: List[str], save_path: str):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(text_data, trainer)
    tokenizer.save(save_path)
    return tokenizer


def train_and_save_BPE_tokenizer(vocab_size: int, text_data: List[str], save_path: str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(text_data, trainer)
    tokenizer.save(save_path)
    return tokenizer


if __name__ == "__main__":
    param_file = sys.argv[1]
    tokenizer_name = sys.argv[2]
    text_file = sys.argv[3]
    queue_name = sys.argv[4]
    save_path = f"../../storage/models/tokenizers/tokenizer_{queue_name}.json"

    with open(param_file) as f:
        params = yaml.safe_load(f)

    vocab_size = params["tokenizer"]["vocab_size"]

    text_data = pd.read_parquet(text_file, columns=["post2_title", "post2_desc", "post1_title", "post1_desc"])
    text_data = (text_data["post2_title"] + " " + text_data["post2_desc"] + " " + text_data["post1_title"] + " " + text_data["post1_desc"]).values
    text_data = [normalize_text(text) for text in text_data]

    if tokenizer_name == "word_piece":
        train_and_save_word_piece_tokenizer(vocab_size=vocab_size, text_data=text_data, save_path=save_path)
    elif tokenizer_name == "BPE":
        train_and_save_BPE_tokenizer(vocab_size=vocab_size, text_data=text_data, save_path=save_path)
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}. It must be one of [word_piece, BPE]")
