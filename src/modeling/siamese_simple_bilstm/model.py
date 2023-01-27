import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from torchmetrics import Accuracy, MetricCollection, AUROC


class BiLSTMBlock(nn.Module):
    def __init__(self,
                 text_emb_dim,
                 num_layers):
        super().__init__()

        self.bilstm_layers = nn.LSTM(
            text_emb_dim, text_emb_dim, bidirectional=True, num_layers=num_layers)

    def forward(self, text_embed, mask=None):
        text_embed, _ = self.bilstm_layers(text_embed)
        text_embed_mean = torch.mean(text_embed, dim=1)
        return text_embed_mean


class DeepBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 text_num_layers,
                 output_feature_dim,
                 dropout_rate: float):
        super().__init__()
        self.save_hyperparameters()

        self.text_embedding = nn.Embedding(text_vocab_size, text_emb_dim)

        self.bilstm = BiLSTMBlock(
            text_emb_dim=text_emb_dim, num_layers=text_num_layers)

        # Title + Desc
        concat_layer_dim = (2 * text_emb_dim) + (2 * text_emb_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.output_layer = nn.Sequential(
            self.dropout_layer,
            nn.Linear(concat_layer_dim, concat_layer_dim),
            nn.ReLU(inplace=True),

            self.dropout_layer,
            nn.Linear(concat_layer_dim, output_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, title_tokens, desc_tokens):
        title_embed = self.text_embedding(title_tokens)
        title_embed_mean = self.bilstm(title_embed)

        desc_embed = self.text_embedding(desc_tokens)
        desc_embed_mean = self.bilstm(desc_embed)

        concated_features = torch.cat(
            [title_embed_mean, desc_embed_mean], dim=1)

        # TODO: try tanh, relu, sigmoid, MultiheadAttention, ELU
        output = torch.tanh(self.output_layer(concated_features))

        return output


class DuplicateSiameseBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 text_num_layers,
                 output_feature_dim,
                 dropout_rate: float,
                 initial_lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = DeepBiLSTM(text_vocab_size,
                                  text_emb_dim,
                                  text_num_layers,
                                  output_feature_dim,
                                  dropout_rate)

        CONCAT_LAYER_SIZE = 2 * output_feature_dim

        self.initial_lr = initial_lr

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.classification_head = nn.Sequential(
            self.dropout_layer,
            nn.Linear(CONCAT_LAYER_SIZE, CONCAT_LAYER_SIZE),

            self.dropout_layer,
            nn.Linear(CONCAT_LAYER_SIZE, 1),

            nn.Sigmoid()
        )

        metrics = MetricCollection([
            Accuracy(task="binary"),
            AUROC(task="binary"),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def _extract_post_data(self, post_x):
        title_tokens, desc_tokens = post_x
        features = {
            "title_tokens": title_tokens,
            "desc_tokens": desc_tokens,
        }
        return features

    def forward(self, post1_x, post2_x):

        post1_input = self._extract_post_data(post1_x)
        post2_input = self._extract_post_data(post2_x)

        post1_feature = self.encoder(**post1_input)
        post2_feature = self.encoder(**post2_input)

        concated_features = torch.cat([post1_feature, post2_feature], dim=1)
        duplicate_score = self.classification_head(concated_features).flatten()

        return duplicate_score

    def training_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

        duplicate_score = self.forward(
            post1_x=post1_x, post2_x=post2_x)

        duplicate_loss = F.binary_cross_entropy(duplicate_score, y_duplicate)

        self.log("duplicate_loss", duplicate_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.train_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True,
                          prog_bar=True, logger=True)

        if (batch_idx != 0) and (batch_idx % 1000 == 0):
            self.lr_schedulers().step()

        return duplicate_loss

    def validation_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

        duplicate_score = self.forward(
            post1_x=post1_x, post2_x=post2_x)

        duplicate_loss = F.binary_cross_entropy(duplicate_score, y_duplicate)

        self.log("val_duplicate_loss", duplicate_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.train_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True,
                          prog_bar=True, logger=True)

        return duplicate_loss

    def predict_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

        duplicate_score = self.forward(
            post1_x=post1_x, post2_x=post2_x)
        return duplicate_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.9, last_epoch=-1, verbose=True)

        return [optimizer], [scheduler]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
