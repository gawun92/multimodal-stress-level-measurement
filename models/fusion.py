import torch
import torch.nn as nn

class LateFeatureFusion(nn.Module):
    def __init__(self, audio_dim=128, face_dim=128, gesture_dim=128, num_classes=3):
        super(LateFeatureFusion, self).__init__()
        
        # Concatenate 3 modality embeddings
        in_features = audio_dim + face_dim + gesture_dim
        
        # 3-layer MLP with Dropout (0.5) and ReLU
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio_emb, face_emb, gesture_emb):
        # Concatenate along feature dimension
        fused = torch.cat([audio_emb, face_emb, gesture_emb], dim=1)
        logits = self.mlp(fused)
        return logits
