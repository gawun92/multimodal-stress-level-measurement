import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = d_model ** 0.5

    def forward(self, x):
        scores = torch.bmm(self.query.expand(x.size(0), -1, -1), x.transpose(1, 2))
        weights = torch.softmax(scores / self.scale, dim=-1)
        pooled = torch.bmm(weights, x).squeeze(1)
        return pooled, weights.squeeze(1)


class FaceBranch(nn.Module):
    def __init__(self, input_dim=286, cnn_channels=(256, 128),
                 lstm_hidden=64, lstm_layers=2, dropout=0.3):
        super().__init__()

        cnn = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn)

        lstm_in = cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.embed_dim = lstm_hidden * 2
        self.attn_pool = AttentionPooling(self.embed_dim)

    def get_sequence(self, x):
        h = x.permute(0, 2, 1)
        h = self.cnn(h)
        h = h.permute(0, 2, 1)
        h, _ = self.lstm(h)
        return h

    def forward(self, x):
        h = self.get_sequence(x)
        emb, _ = self.attn_pool(h)
        return emb


class FaceClassifier(nn.Module):
    def __init__(self, face_branch, num_classes=2, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.backbone = face_branch
        self.head = nn.Sequential(
            nn.Linear(face_branch.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
