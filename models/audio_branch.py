import torch
import torch.nn as nn

class Audio2DCNN(nn.Module):
    def __init__(self, embed_dim=128):
        super(Audio2DCNN, self).__init__()
        # Input shape: (Batch, 1, 128, MaxFrames)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        # x: (B, 1, N_MELS, T)
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        emb = self.fc(x)
        return emb
