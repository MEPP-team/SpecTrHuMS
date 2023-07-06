from learning.modules.CustomLightningModule import CustomLightningModule
from learning.modules.LearnedPooling import LearnedPooling
import utils.welford_means_stds as w
from pytorch_lightning.utilities.model_summary import ModelSummary

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x) if self.training else x


class TransformerEncoder(CustomLightningModule):
    def __init__(self, opt):
        super(TransformerEncoder, self).__init__(opt)

        self.opt = opt

        self.module_static = LearnedPooling(self.opt)

        d_model = self.opt["d_model"]
        nhead = self.opt["n_head"]
        dim_feedforward = self.opt["d_ff"]
        dropout = self.opt["dropout"]
        activation = self.opt["activation_transformer"]

        num_layers = self.opt["num_layers"]

        self.encoder_first_linear = torch.nn.Linear(
            self.opt["size_latent"], d_model).to(opt["device"])

        self.positional_encoding = PositionalEncoding(
            d_model).to(opt["device"])

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        ).to(opt["device"])

        self.encoder_end_linear = torch.nn.Linear(
            d_model, self.opt["size_latent"]).to(opt["device"])

    def forward(self, x):
        unsqueeze = False

        if x.dim() == 3:
            x = x.unsqueeze(0)
            unsqueeze = True

        output_static = self.enc_static(x)

        output_transformer = self.enc_transformer(output_static)

        output = self.dec_static(output_transformer)

        if unsqueeze:
            output = output.squeeze(0)

        return output

    def enc_static(self, x):
        resize = False

        if x.dim() == 4:
            batch_size = x.size(0)
            n_frames = x.size(1)
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.enc(x)

        if resize:
            x = x.view(batch_size, n_frames, -1)

        return x

    def dec_static(self, x):
        resize = False

        if x.dim() == 3:
            batch_size = x.size(0)
            n_frames = x.size(1)
            resize = True
            x = torch.flatten(x, 0, 1)

        x = self.module_static.dec(x)

        if resize:
            x = x.view(batch_size, n_frames, -1, 3)

        return x

    def enc_transformer(self, x):
        # standardize
        x = x - self.means_stds[0]
        x = x / self.means_stds[1]

        start_index = 50

        # last frame repeated
        x[:, start_index:] = x[:, start_index - 1].unsqueeze(1)

        x = self.encoder_first_linear(x)

        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.encoder_end_linear(x)

        # destandardize
        x = x * self.means_stds[1]
        x = x + self.means_stds[0]

        return x
