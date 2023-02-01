import os
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from modeling.siamese_simple_transformer.data import DuplicateDataLoader
from modeling.siamese_simple_transformer.model import DuplicateSiameseTransformer

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
    text_emb_dim = models_params['text_emb_dim']
    text_hidden_dim = models_params['text_hidden_dim']
    text_num_layers = models_params['text_num_layers']
    features_emb_dim = models_params['features_emb_dim']
    output_feature_dim = models_params['output_feature_dim']
    num_heads = models_params['num_heads']
    dropout_rate = models_params['dropout_rate']
    batch_size = models_params['batch_size']
    initial_lr = models_params['initial_lr']


    accumulate_grad_batches = models_params['accumulate_grad_batches']
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

    # torch.set_float32_matmul_precision('medium')

    # Load Model
    if checkpoint_path:
        print("Loading from checkpoint:", checkpoint_path)
        model = DuplicateSiameseTransformer.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        print("Creating New Model.")
        model = DuplicateSiameseTransformer(
            text_vocab_size=data_loader.text_tokenizer.vocab_size,
            slug_vocab_size=len(data_loader.slug_tokenizer.classes_),
            city_vocab_size=len(data_loader.city_tokenizer.classes_),            
            text_emb_dim=text_emb_dim,
            text_num_layers=text_num_layers,
            text_hidden_dim=text_hidden_dim,
            num_heads=num_heads,
            output_feature_dim=output_feature_dim,
            features_emb_dim=features_emb_dim,
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
                                 filename=f"siamese_simple_transformer_{queue_name}_model",
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
                         )

    trainer.fit(model, data_loader)
    return model
