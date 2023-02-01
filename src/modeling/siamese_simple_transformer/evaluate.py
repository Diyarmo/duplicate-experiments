
from modeling.siamese_simple_transformer.model import DuplicateSiameseTransformer
from modeling.siamese_simple_transformer.data import DuplicateDataLoader
import torch
from typing import Tuple
import pytorch_lightning as pl


def get_simple_transformer_model_predictions(model, data_loader):
    predictions = pl.Trainer(accelerator='gpu', gpus=[0], logger=False).predict(
        model, data_loader)
    predictions = torch.sigmoid(torch.cat(predictions, dim=0).type(torch.float)).cpu().numpy().astype("float32")
    return predictions


def get_simple_transformer_model_and_dataloader(
        model_file,
        tokenizer_file,
        slug_tokenizer_file,
        city_tokenizer_file,
        test_path) -> Tuple[DuplicateSiameseTransformer, DuplicateDataLoader]:

    data_loader = DuplicateDataLoader(
        tokenizer_file=tokenizer_file,
        slug_tokenizer_file=slug_tokenizer_file,
        city_tokenizer_file=city_tokenizer_file,        
        batch_size=128,
        text_max_length=512,
        train_file=None,
        test_file=test_path,
        resample_by_label=True).val_dataloader()

    model = DuplicateSiameseTransformer.load_from_checkpoint(
        checkpoint_path=model_file)

    model.eval()

    return model, data_loader
