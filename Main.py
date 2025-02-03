import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # For peak detection

# --------------------
# PARAMETERS
# --------------------
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# --------------------
# UTILITY FUNCTION: CUT SILENCE USING PEAK DETECTION
# --------------------
def cut_silence(audio, peak_height=None, return_indices=False):
    """
    Given a 1D numpy array `audio` and an optional peak_height parameter,
    use SciPy's find_peaks to locate the first and last peaks (based on the absolute amplitude),
    and return the audio slice between them.
    
    Parameters:
      audio (np.ndarray): The input audio signal.
      peak_height (float or None): Minimum height for a peak to be detected.
      return_indices (bool): If True, also return (start, end) indices.
    
    Returns:
      If return_indices is False:
         np.ndarray: The trimmed audio signal.
      Else:
         tuple: (trimmed_audio, start_index, end_index)
         If no peaks are found, the entire audio is returned along with start=0 and end=len(audio).
    """
    peaks, properties = find_peaks(np.abs(audio), height=peak_height)
    if peaks.size == 0:
        if return_indices:
            return audio, 0, len(audio)
        return audio
    start = peaks[0]
    end = peaks[-1] + 1  # +1 so that the last peak is included
    trimmed = audio[start:end]
    if return_indices:
        return trimmed, start, end
    return trimmed

# --------------------
# HELPER FUNCTION: PLOT A SAMPLE WITH CUT INDICES
# --------------------
def plot_cut_silence_sample(original_audio, trimmed_audio, start, end, sample_index):
    """
    Plot the original audio signal and highlight the region kept after cutting.
    The first and last indices (start, end) are marked with vertical dashed lines.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(original_audio, label="Original Audio", color="gray")
    plt.axvline(x=start, color='red', linestyle='--', label="Cut Start" if sample_index==0 else "")
    plt.axvline(x=end, color='red', linestyle='--', label="Cut End" if sample_index==0 else "")
    plt.fill_between(np.arange(start, end), original_audio[start:end], color="lightblue", alpha=0.5,
                     label="Trimmed Region" if sample_index==0 else "")
    plt.title(f"Sample {sample_index+1}: Cut Indices (Start: {start}, End: {end})")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------
# DATASET HANDLING
# --------------------
class CoinDataset(Dataset):
    def __init__(self, file_path, normalize=False, trim_silence_flag=False, silence_threshold=None):
        """
        Parameters:
            file_path (str): Path to the .npy file.
            normalize (bool): Whether to normalize the features.
            trim_silence_flag (bool): Whether to trim silence from the audio.
            silence_threshold (float or None): Used as the minimum peak height.
                                               If None, all peaks are considered.
        """
        print("Loading dataset from:", file_path)
        self.data = np.load(file_path, allow_pickle=True)
        self.labels = self.data[:, 0].astype(int)
        self.features = self.data[:, 1:].astype(float)
        
        self.trim_silence_flag = trim_silence_flag
        self.silence_threshold = silence_threshold

        if normalize:
            print("Normalizing data to [-1, 1] range...")
            self.features = 2 * (self.features - np.min(self.features, axis=1, keepdims=True)) / (
                np.max(self.features, axis=1, keepdims=True) - np.min(self.features, axis=1, keepdims=True) + 1e-7
                ) - 1

        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.features[idx]
        label = self.labels[idx]
        
        if self.trim_silence_flag:
            # For training, we simply return the trimmed audio.
            audio = cut_silence(audio, self.silence_threshold)
        
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --------------------
# ORIGINAL PAD COLLATE FUNCTION (Not used further)
# --------------------
def pad_collate(batch):
    """
    Pads a batch of variable-length tensors with zeros so that they all have the same length.
    """
    audios, labels = zip(*batch)
    max_len = max(audio.size(0) for audio in audios)
    
    padded_audios = []
    for audio in audios:
        pad_size = max_len - audio.size(0)
        if pad_size > 0:
            padded_audio = torch.cat([audio, torch.zeros(pad_size, dtype=audio.dtype)])
        else:
            padded_audio = audio
        padded_audios.append(padded_audio)
    
    return torch.stack(padded_audios), torch.stack(labels)

# --------------------
# NEURAL NETWORK
# --------------------
class CoinClassifier(nn.Module):
    def __init__(self):
        super(CoinClassifier, self).__init__()
        self.initial_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.repeated_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16, track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for _ in range(4)]
        )
        self.final_block = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flattened_size = None
        self.fc = nn.Sequential(
            nn.Linear(1, 100),  # Placeholder; will be reset dynamically.
            nn.BatchNorm1d(100, track_running_stats=False),
            nn.Sigmoid(),
            nn.Linear(100, 7)
        )

    def forward(self, x):
        x = self.initial_block(x)
        x = self.repeated_blocks(x)
        x = self.final_block(x)
        if self.flattened_size is None:
            self.flattened_size = x.shape[1] * x.shape[2]
            self.fc[0] = nn.Linear(self.flattened_size, 100).to(x.device)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.initial_block(x)
        x = self.repeated_blocks(x)
        x = self.final_block(x)
        return x

# --------------------
# HELPER FUNCTION: COMPUTE TSNE FOR CURRENT MODEL REPRESENTATION
# --------------------
def compute_tsne(model, data_loader, device):
    """
    Extract features for all samples in data_loader, run t-SNE on them,
    and return the 2D t-SNE coordinates along with the corresponding labels.
    """
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.unsqueeze(1).to(device)
            feats = model.extract_features(features)
            feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_points = tsne.fit_transform(features_all)
    return tsne_points, labels_all

# --------------------
# TRAINING AND TESTING FUNCTIONS
# --------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, epochs, device):
    print("Starting training...")
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    tsne_results = []  # To store t-SNE results every 10 epochs

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct = 0, 0
        for features, labels in train_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_loss_avg = epoch_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        val_acc = evaluate_model(model, val_loader, device)
        
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss_avg:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        # Every 10 epochs, compute and store t-SNE results on test set.
        if (epoch + 1) % 10 == 0:
            tsne_points, tsne_labels = compute_tsne(model, test_loader, device)
            tsne_results.append({'epoch': epoch+1, 'points': tsne_points, 'labels': tsne_labels})
    
    print("Training completed.")
    return train_losses, train_accuracies, val_accuracies, tsne_results

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)
            outputs = model(features)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(data_loader.dataset)

def compute_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.unsqueeze(1).to(device), labels.to(device)
            outputs = model(features)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# --------------------
# FINAL PLOTTING FUNCTION
# --------------------
def final_plots(train_losses, train_accuracies, val_accuracies, cm, tsne_results):
    # Create a folder "results" if it doesn't exist.
    os.makedirs("results", exist_ok=True)

    # For the composite figure, we use 2 columns overall.
    # Top row: Training Loss and Accuracy (2 subplots)
    # Second row: Confusion Matrix (spanning both columns)
    # Remaining rows: TSNE plots.
    # We now set TSNE grid to have 2 columns.
    tsne_cols = 2
    n_tsne = len(tsne_results)
    tsne_rows = int(np.ceil(n_tsne / tsne_cols))
    total_rows = 2 + tsne_rows  # Row0: Loss & Accuracy, Row1: Confusion Matrix, then TSNE rows

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 5 + 4 * tsne_rows))
    gs = gridspec.GridSpec(total_rows, 2, height_ratios=[1, 1] + [1]*tsne_rows)

    # Training Loss (subplot at (0,0))
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(np.arange(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss')
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    # Training and Validation Accuracy (subplot at (0,1))
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.plot(np.arange(1, len(train_accuracies)+1), train_accuracies, marker='o', label='Train Accuracy')
    ax_acc.plot(np.arange(1, len(val_accuracies)+1), val_accuracies, marker='o', label='Val Accuracy')
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    # Confusion Matrix (subplot spanning row 1, columns 0-1)
    ax_cm = fig.add_subplot(gs[1, :])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["1ct", "2ct", "5ct", "20ct", "50ct", "1€", "2€"])
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
    ax_cm.set_title("Confusion Matrix")

    # TSNE Plots (start at row index 2)
    for i, tsne_dict in enumerate(tsne_results):
        row = 2 + i // tsne_cols
        col = i % tsne_cols
        ax_tsne = fig.add_subplot(gs[row, col])
        points = tsne_dict['points']
        labels = tsne_dict['labels']
        sc = ax_tsne.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', alpha=0.7)
        ax_tsne.set_title(f"t-SNE at Epoch {tsne_dict['epoch']}")
        ax_tsne.set_xlabel("t-SNE Dim 1")
        ax_tsne.set_ylabel("t-SNE Dim 2")
        plt.colorbar(sc, ax=ax_tsne)
    total_tsne_slots = tsne_rows * tsne_cols
    for j in range(len(tsne_results), total_tsne_slots):
        row = 2 + j // tsne_cols
        col = j % tsne_cols
        ax_blank = fig.add_subplot(gs[row, col])
        ax_blank.axis('off')

    plt.tight_layout()
    final_image_path = os.path.join("results", "final_results.png")
    plt.savefig(final_image_path)
    print("Final composite figure saved to:", final_image_path)
    plt.show()

# --------------------
# MAIN FUNCTION
# --------------------
if __name__ == "__main__":
    print("Initializing program...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable silence trimming via peak detection.
    dataset = CoinDataset(file_path="coin_data.npy", normalize=False,
                          trim_silence_flag=True, silence_threshold=10000)
    
    # Compute the fixed length (maximum length among all samples)
    fixed_length = max(len(sample) for sample in dataset.features)
    print(f"Fixed length for padding/truncation: {fixed_length}")

    # Plot the first 5 samples to show cut indices.
    if dataset.trim_silence_flag:
        print("Plotting first 5 samples with cut indices...")
        for i in range(5):
            original_audio = dataset.features[i]
            trimmed_audio, start_idx, end_idx = cut_silence(original_audio, dataset.silence_threshold, return_indices=True)
            plot_cut_silence_sample(original_audio, trimmed_audio, start_idx, end_idx, i)

    print("Splitting dataset...")
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print("Dataset split into train, validation, and test sets.")

    # Define the collate function using the fixed length.
    def fixed_length_collate(batch):
        """
        Pads or truncates each sample in the batch to fixed_length.
        """
        audios, labels = zip(*batch)
        processed_audios = []
        for audio in audios:
            current_length = audio.size(0)
            if current_length < fixed_length:
                pad_size = fixed_length - current_length
                processed_audio = torch.cat([audio, torch.zeros(pad_size, dtype=audio.dtype)])
            else:
                processed_audio = audio[:fixed_length]
            processed_audios.append(processed_audio)
        return torch.stack(processed_audios), torch.stack(labels)
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda batch: fixed_length_collate(batch))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda batch: fixed_length_collate(batch))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda batch: fixed_length_collate(batch))
    print("Data loaders created.")

    print("Initializing model, loss function, and optimizer...")
    model = CoinClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model, loss function, and optimizer initialized.")

    # Train the model and collect history & t-SNE snapshots.
    train_losses, train_accuracies, val_accuracies, tsne_results = train_model(
        model, criterion, optimizer, train_loader, val_loader, test_loader, EPOCHS, device)

    # Compute confusion matrix on the test set.
    cm = compute_confusion_matrix(model, test_loader, device)

    # Generate and display final composite plots and save to file.
    final_plots(train_losses, train_accuracies, val_accuracies, cm, tsne_results)

    print("Program completed.")
