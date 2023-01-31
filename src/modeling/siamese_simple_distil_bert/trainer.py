import os
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from modeling.siamese_simple_distil_bert.data import DuplicateDataLoader
from modeling.siamese_simple_distil_bert.model import DuplicateSiameseBert

queue_name = os.environ["QUEUE_NAME"]


def train_model(
        train_file,
        test_file,
        models_params,
        experiment_name
):

    max_epochs = models_params['max_epochs']
    text_max_length = models_params['text_max_length']
    output_feature_dim = models_params['output_feature_dim']
    dropout_rate = models_params['dropout_rate']
    batch_size = models_params['batch_size']
    initial_lr = models_params['initial_lr']

    accumulate_grad_batches = models_params['accumulate_grad_batches']
    checkpoint_path = models_params['checkpoint_path'] if 'checkpoint_path' in models_params else None
    # Load Data
    data_loader = DuplicateDataLoader(
        batch_size=batch_size,
        text_max_length=text_max_length,
        train_file=train_file,
        test_file=test_file,
        resample_by_label=True
    )

    torch.set_float32_matmul_precision('medium')

    # Load Model
    if checkpoint_path:
        print("Loading from checkpoint:", checkpoint_path)
        model = DuplicateSiameseBert.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        print("Creating New Model.")
        model = DuplicateSiameseBert(
            output_feature_dim=output_feature_dim,
            dropout_rate=dropout_rate,
            initial_lr=initial_lr
        )

    print("Model Parameters Count:", model.count_parameters())
    trainer = pl.Trainer(
        max_epochs=max_epochs,
                         accelerator='gpu', 
                         devices=[0],
                         callbacks=[
                             EarlyStopping(
                                 monitor="val_duplicate_loss", patience=4),

                             ModelCheckpoint(
                                 monitor="val_duplicate_loss",
                                 dirpath="../../storage/models/",
                                 filename=f"siamese_simple_distil_bert_{queue_name}_model",
                                 save_top_k=1,
                                 mode="min"),

                             TQDMProgressBar(refresh_rate=10)
                         ],
                         logger=[
                             TensorBoardLogger(
                                 save_dir="../../logs/tb_logs",
                                 name=experiment_name + "_" + datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 log_graph=True),
                             WandbLogger(
                                 save_dir="../../logs/wandb_logs",
                                 name=experiment_name + "_" + datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 project="DeepDuplicateTransformer",
                                 log_model=False,
                                 offline=True)
                         ],
                         accumulate_grad_batches=accumulate_grad_batches,
                         val_check_interval=0.5,
                         )

    trainer.fit(model, data_loader)
    return model
