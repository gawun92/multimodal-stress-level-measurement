import torch
import torch.nn as nn

class FaceSequenceModel(nn.Module):
    def __init__(self, num_landmarks=478, coords=3, hidden_dim=256, embed_dim=128, num_layers=2):
        super(FaceSequenceModel, self).__init__()
        self.input_size = num_landmarks * coords
        
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, T, 478, 3)
        B, T, L, C = x.size()
        x = x.view(B, T, L * C)  # Flatten spatial dimensions per frame -> (B, T, 478*3)
        
        # lstm_out: (B, T, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output of the last time step
        final_state = lstm_out[:, -1, :]  # (B, hidden_dim)
        emb = self.fc(final_state)
        return emb
