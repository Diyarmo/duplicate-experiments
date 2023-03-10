import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from modeling.siamese_simple_bilstm.data import DuplicateDataLoader
from modeling.siamese_simple_bilstm.model import DuplicateSiameseBiLSTM

queue_name = os.environ["QUEUE_NAME"]


def train_model(
        train_file,
        test_file,
        tokenizer_model_filename,
        slug_tokenizer_file,
        city_tokenizer_file,
        models_params,
        experiment_name
):

    max_epochs = models_params['max_epochs']
    text_max_length = models_params['text_max_length']
    text_embed_dim = models_params['text_embed_dim']
    text_num_layers = models_params['text_num_layers']
    features_emb_dim = models_params['features_emb_dim']
    output_feature_dim = models_params['output_feature_dim']
    dropout_rate = models_params['dropout_rate']
    batch_size = models_params['batch_size']
    initial_lr = models_params['initial_lr']
    checkpoint_path = models_params['checkpoint_path'] if 'checkpoint_path' in models_params else None
    # Load Data
    data_loader = DuplicateDataLoader(
        tokenizer_file=tokenizer_model_filename,
        slug_tokenizer_file=slug_tokenizer_file,
        city_tokenizer_file=city_tokenizer_file,
        batch_size=batch_size,
        text_max_length=text_max_length,
        train_file=train_file,
        test_file=test_file,
        resample_by_label=True
    )

    # Load Model
    if checkpoint_path:
        print("Loading from checkpoint:", checkpoint_path)
        model = DuplicateSiameseBiLSTM.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        print("Creating New Model.")
        model = DuplicateSiameseBiLSTM(
            text_vocab_size=data_loader.text_tokenizer.vocab_size,
            text_emb_dim=text_embed_dim,
            text_num_layers=text_num_layers,
            slug_vocab_size=len(data_loader.slug_tokenizer.classes_),
            city_vocab_size=len(data_loader.city_tokenizer.classes_),
            output_feature_dim=output_feature_dim,
            features_emb_dim=features_emb_dim,
            dropout_rate=dropout_rate,
            initial_lr=initial_lr
        )

    print("Model Parameters Count:", model.count_parameters())
    trainer = pl.Trainer(max_epochs=max_epochs,
                         gpus=1,
                         callbacks=[
                             EarlyStopping(
                                 monitor="val_duplicate_loss", patience=4),

                             ModelCheckpoint(
                                 monitor="val_duplicate_loss",
                                 dirpath="../../storage/models/",
                                 filename=f"siamese_simple_bilstm_{queue_name}_model",
                                 save_top_k=1,
                                 mode="min"),

                            #  TQDMProgressBar(refresh_rate=20)
                         ],
                         logger=[
                             TensorBoardLogger(
                                 save_dir="../../logs/tb_logs",
                                 name=experiment_name + "_" + datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 log_graph=True),
                             WandbLogger(
                                 save_dir="../../logs/wandb_logs",
                                 name=experiment_name + "_" + datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 project="DeepDuplicateBiLSTM",
                                 log_model=False,
                                 offline=True)
                         ],
                         )

    trainer.fit(model, data_loader)
    return model
