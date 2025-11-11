import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import string
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CTCLoss

# Config and hyperparameters
class Config:
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("output")
    MODEL_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    IMG_WIDTH = 200
    IMG_HEIGHT = 50
    
    # settings to tweak
    BATCH_SIZE = 32  #32 -> 64 -> 32
    EPOCHS = 60      # 30 -> 50 -> 60
    VALIDATION_SPLIT = 0.2 
    TEST_SPLIT = 0.1
    INITIAL_LR = 0.001 # .001 -> .005 -> .001
    MIN_LR = 0.00001
    
    @staticmethod
    def create_dirs():
        #Create necessary directories
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        Config.MODEL_DIR.mkdir(exist_ok=True)
        Config.LOGS_DIR.mkdir(exist_ok=True)

# Data loading and preprocessing
class CAPTCHADataLoader:
    def __init__(self, data_dir, img_width=200, img_height=50):
        self.data_dir = Path(data_dir)
        self.img_width = img_width
        self.img_height = img_height
        self.images = []
        self.labels = []
        
    def load_dataset(self):
        print("Loading CAPTCHA dataset...")
        image_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        print(f"Found {len(image_files)} images")

        
        for img_path in tqdm(image_files, desc="Loading images"):
            # get label from filename
            label = img_path.stem
            img = self.load_and_preprocess_image(str(img_path))
            
            if img is not None:
                self.images.append(img)
                self.labels.append(label)
        
        print(f"Successfully loaded {len(self.images)} images")
        return np.array(self.images), self.labels
    
    def load_and_preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Apply preprocessing for model
        img = self.preprocess_image(img)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
            
        return img
    
    def preprocess_image(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0) #reduce noise
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2) #convert to B&W
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #cleaning
        return img

# Character encoding to indices
class CharacterEncoder:
    def __init__(self, characters=None):
        if characters is None:
            characters = string.digits + string.ascii_lowercase
        
        self.characters = sorted(list(set(characters)))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.blank_idx = 0
        self.char_to_idx['<blank>'] = self.blank_idx
        self.idx_to_char[self.blank_idx] = ''
        
    @property
    def num_classes(self): 
        return len(self.characters) + 1
    
    def encode_label(self, label): # convert string label to list of indices
        return [self.char_to_idx[char] for char in label if char in self.char_to_idx]
    
    def decode_label(self, indices): # convert indices back to string
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices if idx != self.blank_idx])
    
    def decode_predictions(self, predictions):
        decoded = []
        if len(predictions.shape) == 3:
            for i in range(predictions.shape[0]):
                pred = torch.argmax(predictions[i], dim=-1)
                decoded.append(self.ctc_decode_single(pred))
        else:
            pred = torch.argmax(predictions, dim=-1)
            decoded.append(self.ctc_decode_single(pred))
        return decoded
    
    def ctc_decode_single(self, prediction):
        decoded = []
        prev_char = None
        
        for char_idx in prediction:
            char_idx = int(char_idx)
            if char_idx == self.blank_idx:
                prev_char = None
                continue
            if char_idx != prev_char:
                decoded.append(char_idx)
                prev_char = char_idx
        
        return self.decode_label(decoded)

# Dataset of Captcha images
class CAPTCHADataset(Dataset):
    def __init__(self, images, labels, encoder):
        self.images = images
        self.labels = labels
        self.encoder = encoder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)
        label = self.labels[idx]
        encoded = self.encoder.encode_label(label)  # Always length 10
        return {
            'image': image,
            'label': torch.LongTensor(encoded),
            'raw_label': label
        }

# collate for batch processing
def collate(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])  # [B, 10]

    # Flatten labels into 1D tensor for CTCLoss
    flat_labels = labels.view(-1)  # shape: B*10

    # Input lengths: width of CNN output after downsampling
    T = images.shape[3] // 8  # adjust if your CNN downsamples differently
    input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
    
    label_lengths = torch.full((images.size(0),), labels.size(1), dtype=torch.long)  
    
    raw_labels = [item['raw_label'] for item in batch]
    
    return {
        'images': images,
        'labels': flat_labels,     # 1D for CTCLoss
        'label_lengths': label_lengths,
        'input_lengths': input_lengths,
        'raw_labels': raw_labels
    }

# make data loaders
def create_data_loaders(train_data, val_data, test_data, encoder, batch_size=32, num_workers=0):
    train_dataset = CAPTCHADataset(train_data[0], train_data[1], encoder)
    val_dataset = CAPTCHADataset(val_data[0], val_data[1], encoder)
    test_dataset = CAPTCHADataset(test_data[0], test_data[1], encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, collate_fn=collate, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Feature extraction using CNN
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        return x

# CRNN using feature extractor
class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, num_rnn_layers=2):
        super(CRNN, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.rnn_input_size = 6 * 256
        
        # Larger LSTM with more dropout
        self.rnn = nn.LSTM(self.rnn_input_size, rnn_hidden_size, num_rnn_layers,
                          batch_first=True, bidirectional=True, dropout=0.3)
        
        # Extra FC layer for better feature extraction
        self.fc1 = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(rnn_hidden_size, num_classes)
        
    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        
        rnn_out, _ = self.rnn(features)
        
        # Additional processing
        output = self.fc1(rnn_out)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output.permute(1, 0, 2)

# CAPTCHA recognition
class CAPTCHARecognizer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def compute_loss(self, outputs, labels, input_lengths, label_lengths):
        log_probs = F.log_softmax(outputs, dim=2)
        return self.criterion(log_probs, labels, input_lengths, label_lengths)

    
    def train_step(self, batch, optimizer):
        self.model.train()
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        label_lengths = batch['label_lengths'].to(self.device)
        
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.compute_loss(outputs, labels, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader, encoder):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                raw_labels = batch['raw_labels']
                
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels, input_lengths, label_lengths)
                total_loss += loss.item()
                
                outputs_np = outputs.permute(1, 0, 2).cpu().numpy()
                predictions = encoder.decode_predictions(torch.tensor(outputs_np))
                
                for pred, true_label in zip(predictions, raw_labels):
                    if pred == true_label:
                        correct += 1
                    total += 1
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

# Save checkpoint for eval
def save_checkpoint(model, optimizer, epoch, accuracy, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': accuracy,
        'val_loss': loss
    }, path)

# Load checkpoint for eval
def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

# Training loop 
def train_model(model, train_loader, val_loader, encoder, config, return_history=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    recognizer = CAPTCHARecognizer(model, device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=config.get('min_lr', 1e-6))

    patience = config.get('early_stopping_patience', 10)
    no_improve = 0
    best_val_accuracy = 0

    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': []
    }

    for epoch in range(config['epochs']):
        epoch_train_losses = []
        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
            for batch in pbar:
                loss = recognizer.train_step(batch, optimizer)
                epoch_train_losses.append(loss)
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = np.mean(epoch_train_losses)
        val_loss, val_accuracy = recognizer.validate(val_loader, encoder)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_accuracy)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}, LR={current_lr:.6f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_accuracy, val_loss, config['model_save_path'])
            print(f"Saved best model with accuracy: {val_accuracy:.4f}")
        else:
            no_improve += 1

        scheduler.step(val_accuracy)

        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    if return_history:
        return best_val_accuracy, history['train_losses'], history['val_losses'], history['val_accuracies']
    return best_val_accuracy


# Seperate into training, validation, and test
def split_dataset(images, labels, val_split=0.2, test_split=0.1):
    X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=test_split, random_state=42)
    val_size = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Evaluate model and get metrics
def evaluate_model(model, test_loader, encoder, device):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    recognizer = CAPTCHARecognizer(model, device)
    
    all_predictions = []
    all_ground_truths = []
    all_confidences = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['label_lengths'].to(device)
            raw_labels = batch['raw_labels']
            
            outputs = model(images)
            
            # Get predictions
            outputs_np = outputs.permute(1, 0, 2).cpu().numpy()
            predictions = encoder.decode_predictions(torch.tensor(outputs_np))
            
            # Calculate confidence
            probs = torch.softmax(outputs, dim=2)
            max_probs, _ = torch.max(probs, dim=2)
            confidences = max_probs.mean(dim=0).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(raw_labels)
            all_confidences.extend(confidences)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(all_predictions, all_ground_truths) if pred == true)
    total = len(all_predictions)
    accuracy = correct / total
    
    # Character-level accuracy
    char_correct = 0
    char_total = 0
    for pred, true in zip(all_predictions, all_ground_truths):
        for i in range(min(len(pred), len(true))):
            if i < len(pred) and pred[i] == true[i]:
                char_correct += 1
            char_total += 1
        char_total += abs(len(pred) - len(true))
    
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    # Collect all errors
    errors = []
    for i, (pred, true) in enumerate(zip(all_predictions, all_ground_truths)):
        if pred != true:
            errors.append({
                'predicted': pred,
                'ground_truth': true,
                'confidence': float(all_confidences[i])
            })
    
    return {
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'errors': errors,
        'average_confidence': float(np.mean(all_confidences))
    }

# Error analysis
def perform_error_analysis(errors):
    if not errors:
        print("No errors to analyze!")
        return
    
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {len(errors)}")
    
    
    # Character confusion analysis
    char_confusions = {}
    
    error_by_position = {i: 0 for i in range(10)}
    
    for error in errors:
        true_label = error['ground_truth']
        pred_label = error['predicted']
        
        # Character-by-character comparison
        for i in range(min(len(true_label), len(pred_label))):
            if i < len(pred_label) and true_label[i] != pred_label[i]:
                confusion_key = f"{true_label[i]} → {pred_label[i]}"
                char_confusions[confusion_key] = char_confusions.get(confusion_key, 0) + 1
                error_by_position[i] = error_by_position.get(i, 0) + 1
    
    # Top character confusions
    print("\nTop 10 character confusions:")
    sorted_confusions = sorted(char_confusions.items(), key=lambda x: x[1], reverse=True)[:10]
    for confusion, count in sorted_confusions:
        print(f"  {confusion}: {count} times")
    
    # Error by position
    '''
    print("\nErrors by character position:")
    for pos, count in sorted(error_by_position.items()):
        if count > 0:
            print(f"  Position {pos}: {count} errors")'''
    
    # Most error-prone characters
    error_by_char = {}
    for error in errors:
        for char in error['ground_truth']:
            error_by_char[char] = error_by_char.get(char, 0) + 1
    
    print("\nMost error-prone characters (top 10):")
    sorted_chars = sorted(error_by_char.items(), key=lambda x: x[1], reverse=True)[:10]
    for char, count in sorted_chars:
        print(f"  '{char}': appears in {count} errors ({count/len(errors)*100:.1f}% of errors)")
    
    # Confidence analysis
    high_conf_errors = [e for e in errors if e['confidence'] > 0.8]
    low_conf_errors = [e for e in errors if e['confidence'] < 0.5]
    
    print(f"\nConfidence analysis:")
    print(f"  High confidence errors (>0.8): {len(high_conf_errors)} ({len(high_conf_errors)/len(errors)*100:.1f}%)")
    print(f"  Low confidence errors (<0.5): {len(low_conf_errors)} ({len(low_conf_errors)/len(errors)*100:.1f}%)")
    print(f"  Average error confidence: {np.mean([e['confidence'] for e in errors]):.3f}")

# Plot training curves
def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark best accuracy
    best_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_acc) + 1
    ax2.plot(best_epoch, best_acc, 'ro', markersize=10)
    ax2.annotate(f'Best: {best_acc:.4f}', 
                xy=(best_epoch, best_acc), 
                xytext=(best_epoch+1, best_acc-0.05),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Show predictions vs. truth
def visualize_predictions(model, test_loader, encoder, device, num_samples=12):
    """Visualize sample predictions"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            raw_labels = batch['raw_labels']
            
            outputs = model(images)
            outputs_np = outputs.permute(1, 0, 2).cpu().numpy()
            predictions = encoder.decode_predictions(torch.tensor(outputs_np))
            
            for i in range(min(len(images), num_samples - len(samples))):
                samples.append({
                    'image': images[i].cpu().squeeze().numpy(),
                    'true': raw_labels[i],
                    'pred': predictions[i]
                })
            
            if len(samples) >= num_samples:
                break
    
    # Plot
    rows = (num_samples + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 3))
    axes = axes.ravel()
    
    for i, sample in enumerate(samples):
        axes[i].imshow(sample['image'], cmap='gray')
        color = 'green' if sample['pred'] == sample['true'] else 'red'
        axes[i].set_title(f"True: {sample['true']}\nPred: {sample['pred']}", color=color, fontsize=10)
        axes[i].axis('off')
    
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'sample_predictions.png')
    plt.show()

#main
if __name__ == "__main__":
    # Setup file struct
    Config.create_dirs()
    
    # Load data
    loader = CAPTCHADataLoader(Config.DATA_DIR, Config.IMG_WIDTH, Config.IMG_HEIGHT)
    images, labels = loader.load_dataset()
    
    # Get character set from data
    all_chars = set()
    for label in labels:
        all_chars.update(label)
    
    # Create encoder with correct character set
    encoder = CharacterEncoder(''.join(sorted(all_chars)))
    print(f"\nEncoder classes (including blank): {encoder.num_classes}")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(images, labels, Config.VALIDATION_SPLIT, Config.TEST_SPLIT)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, encoder,
        batch_size=Config.BATCH_SIZE, num_workers=0
    )
    
    # Create model
    model = CRNN(encoder.num_classes, rnn_hidden_size=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training config
    training_config = {
        'epochs': Config.EPOCHS,
        'learning_rate': Config.INITIAL_LR,
        'min_lr': Config.MIN_LR,
        'early_stopping_patience': 15,
        'model_save_path': Config.MODEL_DIR / 'best_model.pth'
    }
    
    # Train model !
    print("\nStarting training...")
    best_accuracy, train_losses, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, encoder, training_config, return_history=True
    )

    print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies, 
                         Config.OUTPUT_DIR / 'training_curves.png')
    
    # Load best model for evaluation
    checkpoint = load_checkpoint(model, training_config['model_save_path'], device)

    
    # Comprehensive evaluation
    test_results = evaluate_model(model, test_loader, encoder, device)
    
    # Print evaluation metrics
    print(f"\n=== Test Set Results ===")
    print(f"Whole-CAPTCHA Accuracy: {test_results['accuracy']:.4f} ({test_results['correct_predictions']}/{test_results['total_samples']})")
    print(f"Character-level Accuracy: {test_results['char_accuracy']:.4f}")
    print(f"Average Confidence: {test_results['average_confidence']:.4f}")
    print(f"Total Errors: {len(test_results['errors'])}")
    
    # Perform error analysis
    perform_error_analysis(test_results['errors'])
    
    # Visualize sample predictions
    print("\nVisualizing sample predictions...")
    visualize_predictions(model, test_loader, encoder, device)