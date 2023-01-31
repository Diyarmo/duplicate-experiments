import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torchmetrics import Accuracy, MetricCollection, AUROC
from transformers import AutoModel


class PositionalEncoding(pl.LightningModule):

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=(torch.float)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Attention(pl.LightningModule):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()

        self.emb_size = embed_dim
        self.weight = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, mask=None):
        W_bs = self.weight.unsqueeze(0).repeat(x.size()[0], 1, 1) 
        # import pdb; pdb.set_trace()
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


class EncoderBlock(pl.LightningModule):

    def __init__(
        self, 
        text_vocab_size, 
        text_emb_dim, 
        text_hidden_dim, 
        num_layers, 
        dropout_rate, 
        num_heads):
        
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.pos_encoder = PositionalEncoding(d_model=text_emb_dim,
          dropout=dropout_rate,
          vocab_size=text_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=text_emb_dim,
          nhead=num_heads,
          dim_feedforward=text_hidden_dim,
          dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
          num_layers=num_layers)

        self.attention = Attention(text_emb_dim)

    def forward(self, x, mask=None):
        # import pdb; pdb.set_trace()

        x = x * math.sqrt(self.text_emb_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=torch.transpose(mask, 0, 1).type(torch.float))
        # x = x.mean(dim=1)
        x, _ = self.attention(x, mask)

        return x


class Transformer(pl.LightningModule):

    def __init__(
        self, 
        text_vocab_size, 
        text_emb_dim, 
        text_hidden_dim, 
        text_num_layers, 
        num_heads, 
        output_feature_dim, 
        dropout_rate):

        super().__init__()
        self.save_hyperparameters()

        self.text_embedding = nn.Embedding(text_vocab_size, text_emb_dim)
        self.text_encoder = EncoderBlock(
            text_vocab_size=text_vocab_size,
            text_emb_dim=text_emb_dim,
            text_hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            dropout_rate=dropout_rate,
            num_heads=num_heads)


        concat_layer_dim = text_emb_dim
        self.concat_layer = nn.Linear(concat_layer_dim, concat_layer_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(concat_layer_dim, output_feature_dim)

    def forward(self, input_ids, attention_mask):
        text_embed = self.text_embedding(input_ids)
        text_embed_mean = self.text_encoder(text_embed, mask=attention_mask)

        concated_features = torch.cat([text_embed_mean], dim=1)
        concated_features = self.dropout_layer(concated_features)
        concated_features = torch.relu(self.concat_layer(concated_features))
        output = self.output_layer(concated_features)
        return output


class DuplicateSiameseTransformer(pl.LightningModule):
    def __init__(self,
                 text_vocab_size,
                 text_emb_dim,
                 text_hidden_dim,
                 text_num_layers,
                 num_heads,
                 output_feature_dim,
                 dropout_rate: float,
                 initial_lr
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Transformer(
            text_vocab_size, 
            text_emb_dim, 
            text_hidden_dim, 
            text_num_layers, 
            num_heads,
            output_feature_dim, 
            dropout_rate
        )

        CONCAT_LAYER_SIZE = 3 * output_feature_dim

        self.initial_lr = initial_lr

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.classification_head = nn.Sequential(
            self.dropout_layer,
            nn.Linear(CONCAT_LAYER_SIZE, CONCAT_LAYER_SIZE),
            self.dropout_layer,
            nn.Linear(CONCAT_LAYER_SIZE, 1),

        )

        metrics = MetricCollection([
            Accuracy(task="binary"),
            AUROC(task="binary"),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, post1_x, post2_x):
        post1_feature = self.encoder(input_ids=post1_x['input_ids'], attention_mask=post1_x['attention_mask'])
        post2_feature = self.encoder(
            input_ids=post2_x['input_ids'], attention_mask=post2_x['attention_mask'])

        # TODO: Concat both posts and apply attention on that
        concated_features = torch.cat([post1_feature, post2_feature, torch.abs(
            torch.sub(post1_feature, post2_feature))], dim=1)
        duplicate_score = self.classification_head(
            concated_features).flatten()

        return duplicate_score

    def training_step(self, batch, batch_idx):
        post1_x, post2_x, y_duplicate = batch

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

        duplicate_score = self.forward(post1_x=post1_x, post2_x=post2_x)

        return duplicate_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.9, last_epoch=-1, verbose=True)

        return [optimizer], [scheduler]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
