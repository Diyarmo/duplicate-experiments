import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from torchmetrics import Accuracy, MetricCollection, AUROC


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()

        self.emb_size = embed_dim
        self.weight = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, mask=None):
        # Here    x should be [batch_size, time_step, emb_size]
        #      mask should be [batch_size, time_step, 1]

        # Copy the Attention Matrix for batch_size times
        W_bs = self.weight.unsqueeze(0).repeat(x.size()[0], 1, 1)
        # Dot product between input and attention matrix
        scores = torch.bmm(x, W_bs)
        scores = torch.tanh(scores)

        if mask is not None:
            mask = mask.long()
            mask = torch.unsqueeze(mask, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        a_ = torch.softmax(scores.squeeze(-1), dim=-1)
        a = a_.unsqueeze(-1).repeat(1, 1, x.size()[2])

        weighted_input = x * a

        output = torch.sum(weighted_input, dim=1)

        return output, a_


class BiLSTMBlock(nn.Module):
    def __init__(self,
                 text_emb_dim,
                 num_layers):
        super().__init__()

        self.bilstm_layers = nn.LSTM(
            text_emb_dim, text_emb_dim, bidirectional=True, num_layers=num_layers)
        self.attention = Attention(2 * text_emb_dim)

    def forward(self, text_embed, mask=None):
        text_embed, _ = self.bilstm_layers(text_embed)
        text_embed_mean, _ = self.attention(text_embed, mask)
        return text_embed_mean


class DeepBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 text_num_layers,
                 slug_vocab_size,
                 city_vocab_size,
                 features_emb_dim,
                 output_feature_dim,
                 dropout_rate: float):
        super().__init__()
        self.save_hyperparameters()

        self.text_embedding = nn.Embedding(text_vocab_size, text_emb_dim)
        self.slug_embedding = nn.Embedding(slug_vocab_size, features_emb_dim)
        self.city_embedding = nn.Embedding(city_vocab_size, features_emb_dim)

        self.bilstm = BiLSTMBlock(
            text_emb_dim=text_emb_dim, num_layers=text_num_layers)

        # Title + Desc + Slug + City
        concat_layer_dim = (2 * text_emb_dim) + \
            (2 * text_emb_dim) + (2 * features_emb_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.output_layer = nn.Sequential(
            self.dropout_layer,
            nn.Linear(concat_layer_dim, concat_layer_dim),
            nn.ReLU(inplace=True),

            self.dropout_layer,
            nn.Linear(concat_layer_dim, output_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, title_tokens, title_masks, desc_tokens, desc_masks, slug, city):
        title_embed = self.text_embedding(title_tokens)
        title_embed_mean = self.bilstm(title_embed, title_masks)

        desc_embed = self.text_embedding(desc_tokens)
        desc_embed_mean = self.bilstm(desc_embed, desc_masks)

        slug_embed = self.slug_embedding(slug)
        city_embed = self.city_embedding(city)

        concated_features = torch.cat(
            [title_embed_mean, desc_embed_mean, slug_embed, city_embed], dim=1)

        # TODO: try tanh, relu, sigmoid, MultiheadAttention, ELU
        output = torch.tanh(self.output_layer(concated_features))

        return output


class DuplicateSiameseBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 text_num_layers,
                 slug_vocab_size,
                 city_vocab_size,
                 features_emb_dim,
                 output_feature_dim,
                 dropout_rate: float,
                 initial_lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = DeepBiLSTM(text_vocab_size,
                                  text_emb_dim,
                                  text_num_layers,
                                  slug_vocab_size,
                                  city_vocab_size,
                                  features_emb_dim,
                                  output_feature_dim,
                                  dropout_rate)

        CONCAT_LAYER_SIZE = 3 * output_feature_dim

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
        title_tokens, title_masks, desc_tokens, desc_masks, slug, city = post_x
        features = {
            "title_tokens": title_tokens,
            "desc_tokens": desc_tokens,
            "title_masks": title_masks,
            "desc_masks": desc_masks,
            "slug": slug,
            "city": city,
        }
        return features

    def forward(self, post1_x, post2_x):

        post1_input = self._extract_post_data(post1_x)
        post2_input = self._extract_post_data(post2_x)

        post1_feature = self.encoder(**post1_input)
        post2_feature = self.encoder(**post2_input)

        concated_features = torch.cat([post1_feature, post2_feature, torch.abs(
            torch.sub(post1_feature, post2_feature))], dim=1)
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
            output = self.valid_metrics(duplicate_score, y_duplicate.int())
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
