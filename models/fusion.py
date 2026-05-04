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
        return torch.bmm(weights, x).squeeze(1)


class LateFusionClassifier(nn.Module):
    def __init__(self, audio_branch, face_branch, gesture_branch=None,
                 num_classes=2, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.audio_branch = audio_branch
        self.face_branch = face_branch
        self.gesture_branch = gesture_branch

        total_dim = audio_branch.embed_dim + face_branch.embed_dim
        if gesture_branch is not None:
            total_dim += gesture_branch.embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._freeze(self.audio_branch)
        if self.gesture_branch is not None:
            self._freeze(self.gesture_branch)

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, audio_input, face_input, gesture_input=None):
        audio_emb = self.audio_branch(audio_input)
        face_emb = self.face_branch(face_input)

        parts = [audio_emb, face_emb]
        if self.gesture_branch is not None and gesture_input is not None:
            parts.append(self.gesture_branch(gesture_input))

        combined = torch.cat(parts, dim=-1)
        return self.classifier(combined)

    def unfreeze_audio(self):
        for p in self.audio_branch.parameters():
            p.requires_grad = True


class CrossAttentionFusionClassifier(nn.Module):
    def __init__(self, audio_branch, face_branch, gesture_branch=None,
                 embed_dim=128, n_heads=4, num_classes=2,
                 dropout=0.1, classifier_dropout=0.3):
        super().__init__()
        self.ab = audio_branch
        self.fb = face_branch
        self.gb = gesture_branch

        self.attn_a = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_f = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_f = nn.LayerNorm(embed_dim)

        if gesture_branch is not None:
            self.attn_g = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            self.norm_g = nn.LayerNorm(embed_dim)
            self.pool_g = AttentionPooling(embed_dim)

        self.pool_a = AttentionPooling(embed_dim)
        self.pool_f = AttentionPooling(embed_dim)

        out_dim = embed_dim * (3 if gesture_branch is not None else 2)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(classifier_dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        for p in self.ab.parameters(): p.requires_grad = False
        if gesture_branch is not None:
            for p in self.gb.parameters(): p.requires_grad = False

    def forward(self, audio_input, face_input, gesture_input=None):
        a = self.ab.get_sequence(audio_input)
        f = self.fb.get_sequence(face_input)

        has_g = self.gb is not None and gesture_input is not None
        g = self.gb.get_sequence(gesture_input) if has_g else None

        kv_a = torch.cat([f, g], dim=1) if has_g else f
        kv_f = torch.cat([a, g], dim=1) if has_g else a
        a_ctx, _ = self.attn_a(a, kv_a, kv_a); a_ctx = self.norm_a(a_ctx + a)
        f_ctx, _ = self.attn_f(f, kv_f, kv_f); f_ctx = self.norm_f(f_ctx + f)

        parts = [self.pool_a(a_ctx), self.pool_f(f_ctx)]
        if has_g:
            kv_g = torch.cat([a, f], dim=1)
            g_ctx, _ = self.attn_g(g, kv_g, kv_g); g_ctx = self.norm_g(g_ctx + g)
            parts.append(self.pool_g(g_ctx))

        return self.classifier(torch.cat(parts, dim=-1))

    def unfreeze_audio(self):
        for p in self.ab.parameters(): p.requires_grad = True
