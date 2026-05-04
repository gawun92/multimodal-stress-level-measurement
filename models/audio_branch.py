import math
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = d_model ** 0.5

    def forward(self, x):
        attn_scores = torch.bmm(
            self.query.expand(x.size(0), -1, -1), x.transpose(1, 2)
        )
        attn_scores = attn_scores / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, x)
        return pooled.squeeze(1), attn_weights.squeeze(1)


class AudioBranch(nn.Module):
    def __init__(
        self,
        n_mels=128,
        max_frames=1876,
        cnn_channels=None,
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [1, 32, 64, 128]

        self.cnn = nn.Sequential(
            *[
                ConvBlock(cnn_channels[i], cnn_channels[i + 1])
                for i in range(len(cnn_channels) - 1)
            ]
        )

        cnn_out_channels = cnn_channels[-1]
        if cnn_out_channels != embed_dim:
            self.channel_proj = nn.Linear(cnn_out_channels, embed_dim)
        else:
            self.channel_proj = nn.Identity()

        self.pos_enc = SinusoidalPositionalEncoding(
            embed_dim, max_len=500, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attn_pool = AttentionPooling(embed_dim)
        self.embed_dim = embed_dim

    def get_embedding(self, x):
        h = self.cnn(x)
        h = h.mean(dim=2)
        h = h.permute(0, 2, 1)
        h = self.channel_proj(h)
        h = self.pos_enc(h)
        h = self.transformer(h)
        embedding, _ = self.attn_pool(h)
        return embedding

    def forward(self, x):
        return self.get_embedding(x)

    def get_sequence(self, x):
        h = self.cnn(x)
        h = h.mean(dim=2)
        h = h.permute(0, 2, 1)
        h = self.channel_proj(h)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return h

    def forward_with_attention(self, x):
        h = self.cnn(x)
        h = h.mean(dim=2)
        h = h.permute(0, 2, 1)
        h = self.channel_proj(h)
        h = self.pos_enc(h)
        h = self.transformer(h)
        embedding, attn_weights = self.attn_pool(h)
        return embedding, attn_weights


class AudioClassifier(nn.Module):
    def __init__(self, audio_branch, num_classes=2, hidden_dim=64):
        super().__init__()
        self.backbone = audio_branch
        self.head = nn.Sequential(
            nn.Linear(audio_branch.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.head(embedding)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
