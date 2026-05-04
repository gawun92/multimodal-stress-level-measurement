import torch
import torch.nn as nn


class GatedFusionClassifier(nn.Module):
    def __init__(self, ab, fb, gb=None, num_classes=2, hdim=128, drop=0.3):
        super().__init__()
        self.ab, self.fb, self.gb = ab, fb, gb
        n = 2 + (gb is not None)
        d = ab.embed_dim

        self.gate = nn.Sequential(
            nn.Linear(d*n, hdim), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hdim, n), nn.Softmax(dim=-1),
        )
        self.clf = nn.Sequential(
            nn.Linear(d*n, hdim), nn.BatchNorm1d(hdim), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hdim, hdim//2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hdim//2, num_classes),
        )
        for p in ab.parameters(): p.requires_grad = False
        if gb is not None:
            for p in gb.parameters(): p.requires_grad = False

    def forward(self, a, f, gesture_input=None):
        embs = [self.ab(a), self.fb(f)]
        if self.gb is not None and gesture_input is not None:
            embs.append(self.gb(gesture_input))
        cat = torch.cat(embs, dim=-1)
        w   = self.gate(cat)
        return self.clf(torch.cat([e * w[:, i:i+1] for i, e in enumerate(embs)], dim=-1))

    def unfreeze_audio(self):
        for p in self.ab.parameters(): p.requires_grad = True
