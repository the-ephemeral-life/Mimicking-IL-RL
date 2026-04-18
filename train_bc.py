import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py  # <--- Add this line!

# --- 1. DEFINE THE NEURAL NETWORK ---
class BehavioralCloningMLP(nn.Module):
    def __init__(self, input_dim=99, output_dim=29):
        super(BehavioralCloningMLP, self).__init__()
        
        # A simple, fast feed-forward network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Linear output for continuous joint angles (radians)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. DEFINE THE DATASET LOADER ---
class G1ImitationDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}...")
        
        # Handle HDF5 files (.h5)
        if str(filepath).endswith('.h5') or str(filepath).endswith('.hdf5'):
            with h5py.File(filepath, 'r') as f:
                raw_landmarks = f["landmarks"][:]
                self.Y = f["angles"][:].astype(np.float32)
                
        # Handle NumPy compressed files (.npz)
        elif str(filepath).endswith('.npz'):
            data = np.load(filepath, allow_pickle=True)
            raw_landmarks = data["landmarks"]
            self.Y = data["angles"].astype(np.float32)
            
        else:
            raise ValueError("Unsupported file format! Please use .h5 or .npz")

        # Extract features: Landmarks shape (N, 33, 3) -> Flatten to (N, 99)
        self.X = raw_landmarks.reshape(raw_landmarks.shape[0], -1).astype(np.float32)
        
        print(f"Loaded {self.X.shape[0]} frames.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
# --- 3. TRAINING LOOP ---
def train_model(dataset_path="dataset.npz", epochs=50, batch_size=64, lr=0.001):
    # Setup Data
    dataset = G1ImitationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Loss function, and Optimizer
    model = BehavioralCloningMLP()
    criterion = nn.MSELoss() # Mean Squared Error is perfect for continuous regression
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: predict angles from landmarks
            predictions = model(inputs)
            
            # 3. Calculate loss
            loss = criterion(predictions, targets)
            
            # 4. Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print average loss for this epoch
        avg_loss = running_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    # Save the trained brain
    torch.save(model.state_dict(), "G1_bc_brain.pth")
    print("Training complete! Model saved to G1_bc_brain.pth")

if __name__ == "__main__":
    # Change "dataset.npz" to whatever your capture script generated
    train_model(dataset_path="dataset.h5")