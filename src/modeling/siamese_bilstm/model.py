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

        W_bs = self.weight.unsqueeze(0).repeat(x.size()[0], 1, 1)  # Copy the Attention Matrix for batch_size times
        scores = torch.bmm(x, W_bs)  # Dot product between input and attention matrix
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

        self.bilstm_layers = []
        self.LN_layers = []
        for i in range(num_layers):
            if i:
                bilstm_layer = nn.LSTM(2 * text_emb_dim, text_emb_dim, bidirectional=True, num_layers=1)
            else:
                bilstm_layer = nn.LSTM(text_emb_dim, text_emb_dim, bidirectional=True, num_layers=1)

            LN_layer = nn.LayerNorm(2 * text_emb_dim)
            self.bilstm_layers.append(bilstm_layer)
            self.LN_layers.append(LN_layer)

        self.bilstm_layers = nn.ModuleList(self.bilstm_layers)
        self.LN_layers = nn.ModuleList(self.LN_layers)

        self.attention = Attention(2 * text_emb_dim)

    def forward(self, x, mask=None):
        text_embed = x
        layer = 0
        for bilstm_layer, LN_layer in zip(self.bilstm_layers, self.LN_layers):
            prev_text_embed = text_embed
            text_embed, _ = bilstm_layer(text_embed)

            if layer < 1:
                layer += 1
            else:
                text_embed = text_embed + prev_text_embed
            text_embed = LN_layer(text_embed)

        text_embed_mean, _ = self.attention(text_embed, mask)
        return text_embed_mean


class DeepBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 title_num_layers,
                 desc_num_layers,
                 slug_vocab_size: int,
                 city_size: int,
                 neighbor_size: int,
                 features_emb_dim: int,
                 output_feature_dim,
                 dropout_rate: float):
        super().__init__()
        self.save_hyperparameters()

        self.text_embedding = nn.Embedding(text_vocab_size, text_emb_dim)

        self.title_bilstm = BiLSTMBlock(text_emb_dim=text_emb_dim, num_layers=title_num_layers)
        self.desc_bilstm = BiLSTMBlock(text_emb_dim=text_emb_dim, num_layers=desc_num_layers)

        self.slug_embedding = nn.Embedding(slug_vocab_size, features_emb_dim)
        self.city_embedding = nn.Embedding(city_size, features_emb_dim)
        self.neighbor_embedding = nn.Embedding(neighbor_size, features_emb_dim)


        concat_layer_dim = (3 * features_emb_dim) + (2 * text_emb_dim) + (2 * text_emb_dim)  # Title + Desc + cat_slug + city + neighbor
        self.concat_layer = nn.Linear(concat_layer_dim, concat_layer_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(concat_layer_dim, output_feature_dim)

    def forward(self, title_tokens, title_masks, desc_tokens, desc_masks, slug, city, neighbor):
        title_embed = self.text_embedding(title_tokens)
        title_embed_mean = self.title_bilstm(title_embed, mask=title_masks)

        desc_embed = self.text_embedding(desc_tokens)
        desc_embed_mean = self.desc_bilstm(desc_embed, mask=desc_masks)

        slug_embed = self.slug_embedding(slug)
        city_embed = self.city_embedding(city)
        neighbor_embed = self.neighbor_embedding(neighbor)

        concated_features = torch.cat([title_embed_mean, desc_embed_mean, slug_embed, city_embed, neighbor_embed], dim=1)
        concated_features = self.dropout_layer(concated_features)
        concated_features = torch.relu(self.concat_layer(concated_features))

        # TODO: should we change relu to tanh to keep features between (-1, 1)
        output = torch.tanh(self.output_layer(concated_features))

        return output


class DuplicateSiameseBiLSTM(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 title_num_layers,
                 desc_num_layers,
                 slug_vocab_size: int,
                 city_size: int,
                 neighbor_size: int,
                 features_emb_dim: int,
                 output_feature_dim,
                 dropout_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = DeepBiLSTM(text_vocab_size,
                                  text_emb_dim,
                                  title_num_layers,
                                  desc_num_layers,
                                  slug_vocab_size,
                                  city_size,
                                  neighbor_size,
                                  features_emb_dim,
                                  output_feature_dim,
                                  dropout_rate)

        NUM_OF_STATS = 31            
        CONCAT_LAYER_SIZE = 3 * output_feature_dim + NUM_OF_STATS
        self.concat_layer = nn.Linear(CONCAT_LAYER_SIZE, CONCAT_LAYER_SIZE)
        self.mixing_layer = nn.Linear(CONCAT_LAYER_SIZE, CONCAT_LAYER_SIZE)
        self.classification_head = nn.Linear(CONCAT_LAYER_SIZE, 1)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        self.dropout_layer = nn.Dropout(dropout_rate)


        metrics = MetricCollection([
            Accuracy(task="binary"),
            AUROC(task="binary"),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def _extract_post_data(self, post_x):
        title_tokens, title_masks, desc_tokens, desc_masks, slug, city, neighbor = post_x
        features = {
            "title_tokens": title_tokens,
            "title_masks": title_masks,
            "desc_tokens": desc_tokens,
            "desc_masks": desc_masks,
            "slug": slug,
            "city": city,
            "neighbor": neighbor
                    }
        return features

    def forward(self, post1_x, post2_x, stats):

        post1_input = self._extract_post_data(post1_x)
        post2_input = self._extract_post_data(post2_x)

        post1_feature = self.encoder(**post1_input)
        post2_feature = self.encoder(**post2_input)

        concated_features = torch.cat([post1_feature, post2_feature, torch.abs(torch.sub(post1_feature, post2_feature)), stats], dim=1)
        concated_features = self.dropout_layer(concated_features)
        concated_features = self.concat_layer(concated_features)
        concated_features = self.mixing_layer(concated_features)
        concated_features = self.classification_head(concated_features)
        duplicate_score = torch.sigmoid(concated_features).flatten()

        return post1_feature, post2_feature, duplicate_score

    def training_step(self, batch, batch_idx):
        post1_x, post2_x, stats, y_duplicate = batch

        post1_features, post2_features, duplicate_score = self.forward(post1_x=post1_x, post2_x=post2_x, stats=stats)

        duplicate_loss = F.binary_cross_entropy(duplicate_score, y_duplicate)
        contrastive_loss = self.contrastive_loss(post1_features, post2_features, y_duplicate)
        total_loss = duplicate_loss + contrastive_loss

        self.log("duplicate_loss", duplicate_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.train_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if (batch_idx != 0) and (batch_idx % 1000 == 0):
            self.lr_schedulers().step()

        return duplicate_loss

    def validation_step(self, batch, batch_idx):
        post1_x, post2_x, stats, y_duplicate = batch

        post1_features, post2_features, duplicate_score = self.forward(post1_x=post1_x, post2_x=post2_x, stats=stats)

        duplicate_loss = F.binary_cross_entropy(duplicate_score, y_duplicate)
        contrastive_loss = self.contrastive_loss(post1_features, post2_features, y_duplicate)
        total_loss = duplicate_loss + contrastive_loss

        self.log("val_duplicate_loss", duplicate_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.valid_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return duplicate_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1, verbose=True)

        return [optimizer], [scheduler]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# TODO: check this https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
