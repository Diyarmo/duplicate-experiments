import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torchmetrics import Accuracy, MetricCollection, AUROC
from transformers import AutoModel


class Attention(pl.LightningModule):
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
            scores = scores.masked_fill(mask == 0, torch.float16(-1e9))

        a_ = torch.softmax(scores.squeeze(-1), dim=-1)
        a = a_.unsqueeze(-1).repeat(1, 1, x.size()[2])

        weighted_input = x * a

        output = torch.sum(weighted_input, dim=1)

        return output, a_


class DuplicateSiameseBert(pl.LightningModule):
    def __init__(self,
                 output_feature_dim,
                 dropout_rate: float,
                 initial_lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(
            "HooshvareLab/distilbert-fa-zwnj-base")

        for p in self.encoder.parameters():
            p.requires_grad = False

        CONCAT_LAYER_SIZE = 3 * output_feature_dim

        self.initial_lr = initial_lr
        self.attenion = Attention(output_feature_dim)
        # self.MHA = nn.MultiheadAttention(
        #     output_feature_dim, num_heads=2, dropout=0.2)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.classification_head = nn.Sequential(
            self.dropout_layer,
            nn.Linear(CONCAT_LAYER_SIZE, output_feature_dim),
            self.dropout_layer,
            nn.Linear(output_feature_dim, 1),

        )

        metrics = MetricCollection([
            Accuracy(task="binary"),
            AUROC(task="binary"),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, post1_x, post2_x):
        # import pdb; pdb.set_trace()

        post1_feature = self.encoder(
            input_ids=post1_x['input_ids'], attention_mask=post1_x['attention_mask'])['last_hidden_state']
        post2_feature = self.encoder(
            input_ids=post2_x['input_ids'], attention_mask=post2_x['attention_mask'])['last_hidden_state']

        post1_feature, _ = self.attenion(post1_feature)
        post2_feature, _ = self.attenion(post2_feature)

        # TODO: Concat both posts and apply attention on that
        concated_features = torch.cat([post1_feature, post2_feature, torch.abs(
            torch.sub(post1_feature, post2_feature))], dim=1)
        duplicate_score = self.classification_head(
            concated_features).flatten()

        return duplicate_score

    def training_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch
        # import pdb; pdb.set_trace()
        with autocast():
            duplicate_score = self.forward(
                post1_x=post1_x, post2_x=post2_x)

            duplicate_loss = F.binary_cross_entropy_with_logits(duplicate_score, y_duplicate)

        self.log("duplicate_loss", duplicate_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.train_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True,
                          prog_bar=True, logger=True)

        if (batch_idx != 0) and (batch_idx % 4000 == 0):
            self.lr_schedulers().step()

        return duplicate_loss

    def validation_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

        with autocast():
            duplicate_score = self.forward(
                post1_x=post1_x, post2_x=post2_x)

            duplicate_loss = F.binary_cross_entropy_with_logits(duplicate_score, y_duplicate)


        self.log("val_duplicate_loss", duplicate_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 20 == 0:
            output = self.valid_metrics(duplicate_score, y_duplicate.int())
            self.log_dict(output, on_step=True, on_epoch=True,
                          prog_bar=True, logger=True)

        return duplicate_loss

    def predict_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

        with autocast():
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
