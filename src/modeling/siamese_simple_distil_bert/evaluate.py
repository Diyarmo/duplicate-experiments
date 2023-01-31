
from modeling.siamese_simple_distil_bert.model import DuplicateSiameseBert
from modeling.siamese_simple_distil_bert.data import DuplicateDataLoader
import torch
from typing import Tuple
import pytorch_lightning as pl


def get_simple_distil_bert_model_predictions(model, data_loader):
    predictions = pl.Trainer(accelerator='gpu', gpus=[0], logger=False).predict(
        model, data_loader)
    predictions = torch.sigmoid(torch.cat(predictions, dim=0).type(torch.float)).cpu().numpy().astype("float32")
    return predictions


def get_simple_distil_bert_model_and_dataloader(
        model_file,
        test_path) -> Tuple[DuplicateSiameseBert, DuplicateDataLoader]:

    data_loader = DuplicateDataLoader(
        batch_size=128,
        text_max_length=None,
        train_file=None,
        test_file=test_path,
        resample_by_label=True).val_dataloader()

    model = DuplicateSiameseBert.load_from_checkpoint(
        checkpoint_path=model_file)

    model.eval()

    return model, data_loader
