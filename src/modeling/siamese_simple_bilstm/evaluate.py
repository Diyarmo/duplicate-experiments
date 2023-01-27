
from modeling.siamese_simple_bilstm.model import DuplicateSiameseBiLSTM
from modeling.siamese_simple_bilstm.data import DuplicateDataLoader
import torch
from tqdm import tqdm
from typing import Tuple
import numpy as np
import pytorch_lightning as pl


def get_simple_bilstm_model_predictions(model, data_loader):
    predictions = pl.Trainer(accelerator='gpu', gpus=[1]).predict(
        model, data_loader)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    return predictions


def get_simple_bilstm_model_and_dataloader(
        model_file,
        tokenizer_file,
        test_path) -> Tuple[DuplicateSiameseBiLSTM, DuplicateDataLoader]:

    data_loader = DuplicateDataLoader(
        tokenizer_file=tokenizer_file,
        batch_size=256,
        text_max_length=None,
        train_file=None,
        test_file=test_path,
        test_sample_size=200000).val_dataloader()

    model = DuplicateSiameseBiLSTM.load_from_checkpoint(
        checkpoint_path=model_file)

    model.eval()

    return model, data_loader
