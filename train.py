import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Import our models
from models.audio_branch import Audio2DCNN
from models.face_branch import FaceSequenceModel
from models.gesture_branch import Gesture1DCNN
from models.fusion import LateFeatureFusion
from dataset import StressIDDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Example hyperparameters
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4

    # Load dataset
    # Dynamically select path based on execution environment
    CARC_DATA_DIR = "/project2/msoleyma_1026/group_14"
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_DATA_DIR = BASE_DIR
    root_dir = CARC_DATA_DIR if os.path.exists(os.path.join(CARC_DATA_DIR, "data", "stressid")) else LOCAL_DATA_DIR
    
    train_data = StressIDDataset(root_dir=root_dir, split="train")
    
    # We create a dummy sampler for class imbalance, demonstrating WeightedRandomSampler
    # Normally, you would compute weights based on actual class frequencies.
    # We assume 'labels' represents the 3 classes.
    if len(train_data) > 0:
        class_sample_count = np.array([1, 1, 1]) # placeholder for num classes counts
        weight = 1. / class_sample_count
        # This iterates over the mocked dataset to find weights. 
        # Needs true labels loaded in Dataset for real use.
        samples_weight = np.array([weight[t["label"].item()] for t in train_data])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    else:
        # Fallback if no data is found locally during dry-run testing
        train_loader = []

    # Initialize all models
    audio_net = Audio2DCNN(embed_dim=128).to(device)
    face_net = FaceSequenceModel(embed_dim=128).to(device)
    gesture_net = Gesture1DCNN(embed_dim=128).to(device)
    fusion_net = LateFeatureFusion(audio_dim=128, face_dim=128, gesture_dim=128, num_classes=3).to(device)

    # Collect parameters from all branches for a single optimizer
    params = list(audio_net.parameters()) + list(face_net.parameters()) + \
             list(gesture_net.parameters()) + list(fusion_net.parameters())
    
    optimizer = optim.Adam(params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("[START] Multi-modal Training Loop")
    for epoch in range(epochs):
        audio_net.train()
        face_net.train()
        gesture_net.train()
        fusion_net.train()
        
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            gesture = batch["gesture"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass per modality
            audio_emb = audio_net(audio)
            face_emb = face_net(face)
            gesture_emb = gesture_net(gesture)
            
            # Late feature level fusion
            logits = fusion_net(audio_emb, face_emb, gesture_emb)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/max(1, len(train_loader)):.4f}")
        
    print("[END] Training Complete")

if __name__ == "__main__":
    train_model()
