import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 75
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# --------------------
# DATASET HANDLING
# --------------------
class CoinDataset(Dataset):
    def __init__(self, file_path, normalize=False):
        print("Loading dataset from:", file_path)
        # Load the data from the .npy file
        self.data = np.load(file_path, allow_pickle=True)
        self.labels = self.data[:, 0].astype(int)  # First column is the label
        self.features = self.data[:, 1:].astype(float)  # Remaining columns are the features

        if normalize:
            print("Normalizing data...")
            # Normalize data between 0 and 1
            self.features = (self.features - np.min(self.features, axis=1, keepdims=True)) / (
                np.max(self.features, axis=1, keepdims=True) - np.min(self.features, axis=1, keepdims=True) + 1e-7
            )
        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --------------------
# NEURAL NETWORK
# --------------------
class CoinClassifier(nn.Module):
    def __init__(self):
        super(CoinClassifier, self).__init__()
        
        # Initial Block (Red Arrow)
        self.initial_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 4x Repeated Conv+Pool Blocks (Red + Green Arrows)
        self.repeated_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16, track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for _ in range(4)]
        )
        
        # Final Conv+Pool Block (Red + Green Arrows)
        self.final_block = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Dynamically calculate flattened size
        self.flattened_size = None

        # FC Layers (Pink Arrows)
        self.fc = nn.Sequential(
            nn.Linear(1, 100),  # Placeholder (updated dynamically)
            nn.BatchNorm1d(100, track_running_stats=False),
            nn.Sigmoid(),
            nn.Linear(100, 7)
        )

    def forward(self, x):
        # Input shape: (batch, 1, input_length)
        x = self.initial_block(x)
        x = self.repeated_blocks(x)
        x = self.final_block(x)
        
        # Dynamically adjust FC layer
        if self.flattened_size is None:
            self.flattened_size = x.shape[1] * x.shape[2]  # channels * seq_length
            self.fc[0] = nn.Linear(self.flattened_size, 100).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# --------------------
# TRAINING AND TESTING FUNCTIONS
# --------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    print("Starting training...")
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for features, labels in train_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)  # Add channel dimension and move to device
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        val_acc = evaluate_model(model, val_loader)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    print("Training completed.")

    # Plot training loss and accuracy trends
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Trend')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Trend')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the final model
    torch.save(model.state_dict(), "coin_classifier_model.pth")
    print("Model saved to coin_classifier_model.pth")

def evaluate_model(model, data_loader):
    print("Evaluating model...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)  # Move to device
            outputs = model(features)
            correct += (outputs.argmax(1) == labels).sum().item()
    print("Evaluation completed.")
    return correct / len(data_loader.dataset)

def test_model(model, test_loader):
    print("Testing model...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)  # Move to device
            outputs = model(features)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1ct", "2ct", "5ct", "20ct", "50ct", "1€", "2€"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    print("Testing completed.")

# --------------------
# MAIN FUNCTION
# --------------------
if __name__ == "__main__":
    print("Initializing program...")
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    dataset = CoinDataset(file_path="coin_data.npy", normalize=False)

    # Split the dataset
    print("Splitting dataset...")
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print("Dataset split into train, validation, and test sets.")

    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaders created.")

    # Initialize the model, loss function, and optimizer
    print("Initializing model, loss function, and optimizer...")
    model = CoinClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model, loss function, and optimizer initialized.")

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)

    # Test the model
    test_model(model, test_loader)
    print("Program completed.")
