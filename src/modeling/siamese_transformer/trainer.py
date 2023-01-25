from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modeling.siamese_transformer.data import DuplicateDataLoader
from modeling.siamese_transformer.model import DuplicateSiameseTransformer

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
    text_hidden_dim = models_params['text_hidden_dim']
    text_num_layers = models_params['text_num_layers']
    num_heads = models_params['num_heads']
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
        test_sample_size=100000,
    )

    # Load Model
    if checkpoint_path:
        print("Loading from checkpoint:", checkpoint_path)
        model = DuplicateSiameseTransformer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
        )
    else:
        print("Create new model.")
        model = DuplicateSiameseTransformer(
            text_vocab_size=data_loader.text_tokenizer.vocab_size,
            text_emb_dim=text_embed_dim,
            text_num_layers=text_num_layers,
            text_hidden_dim=text_hidden_dim,
            num_heads=num_heads,
            slug_vocab_size=len(data_loader.slug_tokenizer.classes_),
            city_size=len(data_loader.city_tokenizer.classes_),
            neighbor_size=len(data_loader.neighbor_tokenizer.classes_),
            features_emb_dim=features_embedding_dim,
            output_feature_dim=output_feature_dim,
            dropout_rate=dropout_rate,
        )

    print("Model Parameters Count:", model.count_parameters())
    trainer = pl.Trainer(max_epochs=max_epochs,
                         # progress_bar_refresh_rate=2,
                         gpus=1,
                         callbacks=[
                             EarlyStopping(
                                 monitor="val_total_loss", patience=4),
                                    ModelCheckpoint(
                                        monitor="val_total_loss",
                                        dirpath="../../storage/models/",
                                        filename=f"siamese-transformer-model-{queue_name}",
                                        save_top_k=1,
                                        mode="min")
                                    ],
                         # checkpoint_callback=True,
                         logger=TensorBoardLogger(
                             save_dir="../../logs/tb_logs",
                             name=datetime.now().strftime('%Y-%m-%d--%H-%M'),
                             log_graph=True),
                         # weights_summary="full"
                         )

    trainer.fit(model, data_loader)
    return model
