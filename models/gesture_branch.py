"""
models/gesture_branch.py

Temporal sequence models for landmark-based motion inputs.

Default behavior matches the original hand-gesture pipeline:
    per-frame CNN -> BiLSTM -> attention pooling

When `joint_count` is provided, the branch switches to a pose-aware mode:
    per-joint MLP -> joint self-attention -> joint attention pooling per frame
    -> BiLSTM -> temporal attention pooling
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learnable query vector that attends over the temporal sequence."""

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = d_model ** 0.5

    def forward(self, x, mask=None):
        attn_scores = torch.bmm(
            self.query.expand(x.size(0), -1, -1), x.transpose(1, 2)
        )
        attn_scores = attn_scores / self.scale

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)
            all_invalid = ~mask.any(dim=1)
            if all_invalid.any():
                mask = mask.clone()
                mask[all_invalid] = True
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, x)
        return pooled.squeeze(1), attn_weights.squeeze(1)


class JointAttentionPooling(nn.Module):
    """Learnable query vector that pools joint embeddings within one frame."""

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


class GestureBranch(nn.Module):
    """
    Landmark branch with two modes:
        1. Legacy flat-vector mode for hand landmarks
        2. Pose-aware mode for structured body-pose landmarks
    """

    def __init__(
        self,
        input_size=63,
        cnn_channels=None,
        kernel_size=3,
        embed_dim=128,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
        joint_count=None,
        coord_dim=3,
        joint_embed_dim=64,
        joint_attn_heads=4,
    ):
        super().__init__()
        self.joint_count = joint_count
        self.coord_dim = coord_dim
        self.pose_aware = joint_count is not None

        if self.pose_aware:
            expected_size = joint_count * coord_dim
            if input_size != expected_size:
                raise ValueError(
                    f"Pose-aware mode expects input_size={expected_size} for "
                    f"joint_count={joint_count}, coord_dim={coord_dim}, got {input_size}."
                )
            self.joint_encoder = nn.Sequential(
                nn.Linear(coord_dim, joint_embed_dim),
                nn.LayerNorm(joint_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(joint_embed_dim, joint_embed_dim),
                nn.LayerNorm(joint_embed_dim),
                nn.ReLU(),
            )
            self.joint_attn = nn.MultiheadAttention(
                embed_dim=joint_embed_dim,
                num_heads=joint_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.joint_pool = JointAttentionPooling(joint_embed_dim)
            frame_feature_dim = joint_embed_dim
        else:
            if cnn_channels is None:
                cnn_channels = [128, 64]

            cnn_layers = []
            in_ch = input_size
            for out_ch in cnn_channels:
                cnn_layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = out_ch
            self.cnn = nn.Sequential(*cnn_layers)
            frame_feature_dim = cnn_channels[-1]

        self.lstm = nn.LSTM(
            input_size=frame_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn_pool = AttentionPooling(lstm_hidden * 2)

        lstm_out_dim = lstm_hidden * 2
        if lstm_out_dim != embed_dim:
            self.proj = nn.Linear(lstm_out_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        self.embed_dim = embed_dim

    def _encode_frames(self, x):
        B, T, F = x.shape
        if self.pose_aware:
            joints = x.view(B * T, self.joint_count, self.coord_dim)
            joints = self.joint_encoder(joints)
            joints, _ = self.joint_attn(joints, joints, joints, need_weights=False)
            frames, _ = self.joint_pool(joints)
            return frames.view(B, T, -1)

        h = x.view(B * T, F, 1)
        h = self.cnn(h).mean(dim=-1)
        return h.view(B, T, -1)

    def get_embedding(self, x, mask=None):
        h = self._encode_frames(x)
        h, _ = self.lstm(h)

        if mask is not None:
            mask = mask.to(device=h.device, dtype=torch.bool)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)

        embedding, _ = self.attn_pool(h, mask=mask)
        embedding = self.proj(embedding)
        return embedding

    def forward(self, x, mask=None):
        return self.get_embedding(x, mask=mask)

    def forward_with_attention(self, x, mask=None):
        h = self._encode_frames(x)
        h, _ = self.lstm(h)

        if mask is not None:
            mask = mask.to(device=h.device, dtype=torch.bool)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)

        embedding, attn_weights = self.attn_pool(h, mask=mask)
        embedding = self.proj(embedding)
        return embedding, attn_weights


class GestureClassifier(nn.Module):
    def __init__(self, gesture_branch, num_classes=2, hidden_dim=64):
        super().__init__()
        self.backbone = gesture_branch
        self.head = nn.Sequential(
            nn.Linear(gesture_branch.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        embedding = self.backbone(x, mask=mask)
        return self.head(embedding)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
