import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from modeling.siamese_bilstm.data import DuplicateDataLoader
from modeling.siamese_bilstm.model import DuplicateSiameseBiLSTM

queue_name = os.environ["QUEUE_NAME"]

def train_model(
        train_file,
        test_file,
        tokenizer_model_filename,
        slug_tokenizer_model_filename,
        city_tokenizer_model_filename,
        neighbor_tokenizer_model_filename,
        models_params
):

    max_epochs = models_params['max_epochs']
    text_max_length = models_params['text_max_length']
    text_embed_dim = models_params['text_embed_dim']
    title_num_layers = models_params['title_lstml_num_layers']
    desc_num_layers = models_params['desc_lstm_num_layers']
    features_embedding_dim = models_params['features_embedding_dim']
    output_feature_dim = models_params['output_feature_dim']
    dropout_rate = models_params['dropout_rate']
    batch_size = models_params['batch_size']
    checkpoint_path = models_params['checkpoint_path'] if 'checkpoint_path' in models_params else None
    # Load Data
    data_loader = DuplicateDataLoader(
        tokenizer_file=tokenizer_model_filename,
        slug_tokenizer_file=slug_tokenizer_model_filename,
        city_tokenizer_file=city_tokenizer_model_filename,
        neighbor_tokenizer_file=neighbor_tokenizer_model_filename,
        batch_size=batch_size,
        text_max_length=text_max_length,
        train_file=train_file,
        test_file=test_file,
        test_sample_size=200000
    )

    # Load Model
    if checkpoint_path:
        print("Loading from checkpoint:", checkpoint_path)
        model = DuplicateSiameseBiLSTM.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            text_vocab_size=data_loader.text_tokenizer.vocab_size,
            text_emb_dim=text_embed_dim,
            title_num_layers=title_num_layers,
            desc_num_layers=desc_num_layers,
            slug_vocab_size=len(data_loader.slug_tokenizer.classes_),
            city_size=len(data_loader.city_tokenizer.classes_),
            neighbor_size=len(data_loader.neighbor_tokenizer.classes_),
            features_emb_dim=features_embedding_dim,
            output_feature_dim=output_feature_dim,
            dropout_rate=dropout_rate,
        )
    else:
        print("Create new model.")
        model = DuplicateSiameseBiLSTM(
            text_vocab_size=data_loader.text_tokenizer.vocab_size,
            text_emb_dim=text_embed_dim,
            title_num_layers=title_num_layers,
            desc_num_layers=desc_num_layers,
            slug_vocab_size=len(data_loader.slug_tokenizer.classes_),
            city_size=len(data_loader.city_tokenizer.classes_),
            neighbor_size=len(data_loader.neighbor_tokenizer.classes_),
            features_emb_dim=features_embedding_dim,
            output_feature_dim=output_feature_dim,
            dropout_rate=dropout_rate,
        )

    print("Model Parameters Count:", model.count_parameters())
    trainer = pl.Trainer(max_epochs=max_epochs,
                         gpus=1,
                         callbacks=[
                            EarlyStopping(monitor="val_duplicate_loss", patience=4),
                            
                            ModelCheckpoint(
                                monitor="val_duplicate_loss",
                                dirpath="../../storage/models/",
                                filename=f"siamese-bilstm-model-{queue_name}",
                                save_top_k=1,
                                mode="min"),
                            
                            TQDMProgressBar(refresh_rate=50) 
                            ],
                         # checkpoint_callback=True,
                         logger=[
                             TensorBoardLogger(
                                 save_dir="../../logs/tb_logs",
                                 name=datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 log_graph=True),
                             WandbLogger(
                                 save_dir="../../logs/wandb_logs",
                                 name=datetime.now().strftime('%Y-%m-%d--%H-%M'),
                                 project="DeepDuplicateBiLSTM",
                                 log_model=False,
                                 offline=True)
                             ],
                         # weights_summary="full"
                         )

    trainer.fit(model, data_loader)
    return model
