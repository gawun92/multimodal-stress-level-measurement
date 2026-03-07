import torch
import torch.nn as nn

class Gesture1DCNN(nn.Module):
    def __init__(self, num_landmarks=21, coords=3, embed_dim=128):
        super(Gesture1DCNN, self).__init__()
        self.in_channels = num_landmarks * coords  # 21 * 3 = 63 features per timestep
        
        # Input shape expected: (Batch, Channels, Time) => (B, 63, T)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, embed_dim)
        
    def forward(self, x):
        # x: (B, T, 21, 3)
        B, T, L, C = x.size()
        x = x.view(B, T, L * C)   # (B, T, 63)
        x = x.permute(0, 2, 1)    # (B, channels=63, time=T) for Conv1d
        
        x = self.conv_blocks(x)
        x = self.global_pool(x)   # (B, 256, 1)
        x = x.squeeze(-1)         # (B, 256)
        
        emb = self.fc(x)
        return emb
