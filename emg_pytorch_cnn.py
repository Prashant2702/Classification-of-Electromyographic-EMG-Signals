import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
import joblib
import os
from typing import Tuple, Optional

class EMG_CNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        """Initialize EMG CNN classifier
        
        Args:
            input_shape: Shape of input data (features, 1)
            num_classes: Number of output classes
        """
        super(EMG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * (input_shape[0]//4), 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EMG_Classifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)

    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess EMG data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"EMG data file not found: {filepath}")
            
        df = pd.read_csv("emg_data/EMG-data.csv")  # Update to new path
        X = df.iloc[:, :-1].values  # Keep existing logic
        y = df.iloc[:, -1].values
        
        # Normalize and reshape for CNN
        X = self.scaler.fit_transform(X)
        X = X.reshape(X.shape[0], 1, X.shape[1])  # (batch, channels, features)
        
        # Encode labels
        y = self.le.fit_transform(y)
        
        return X, y

    def train(self, filepath: str, epochs: int = 10, batch_size: int = 32):
        """Train CNN model"""
        X, y = self.load_data(filepath)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create model
        self.model = EMG_CNN((X.shape[2], 1), len(np.unique(y))).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Train/validation split
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                # Calculate metrics
                all_preds = []
                all_labels = []
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                f1 = f1_score(all_labels, all_preds, average='weighted')
                cm = confusion_matrix(all_labels, all_preds)
                
                # Calculate specificity
                tn = cm.diagonal()
                fp = cm.sum(axis=0) - tn
                specificity = tn / (tn + fp)
                
                print(f'Epoch {epoch+1}/{epochs}, Validation Metrics:')
                print(f'Accuracy: {accuracy*100:.2f}%')
                print(f'Precision: {precision:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'Specificity: {np.mean(specificity):.4f}')
                print(f'F1 Score: {f1:.4f}')
        
        # Save model and preprocessing objects
        torch.save(self.model.state_dict(), 'models/emg_cnn_model.pth')  # Update to new path
        joblib.dump(self.scaler, 'models/cnn_scaler.pkl')  # Update to new path
        joblib.dump(self.le, 'models/label_encoder.pkl')  # Update to new path

    def predict(self, filepath: str, return_metrics: bool = False) -> np.ndarray:
        """Make predictions on new EMG data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prediction file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        X = df.values
        X = self.scaler.transform(X)
        X = X.reshape(X.shape[0], 1, X.shape[1])  # (batch, channels, features)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            preds = self.le.inverse_transform(predicted.cpu().numpy())
            
            if return_metrics:
                # For prediction metrics, we'd need true labels which aren't available
                # So we just return basic prediction counts
                unique, counts = np.unique(preds, return_counts=True)
                pred_counts = dict(zip(unique, counts))
                
                return {
                    'predictions': preds,
                    'prediction_counts': pred_counts
                }
            return preds

if __name__ == "__main__":
    # Create required directories if they don't exist
    os.makedirs("emg_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Initializing EMG Classifier...")
    classifier = EMG_Classifier()
    
    try:
        print("\nTraining model...")
        train_file = "emg_data/EMG-data.csv"
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data file not found at: {os.path.abspath(train_file)}")
            
        classifier.train(train_file)
        
        print("\nMaking predictions...")
        test_file = "emg_data/test-data.csv"
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data file not found at: {os.path.abspath(test_file)}")
            
        predictions = classifier.predict(test_file, return_metrics=True)
        print("\nPrediction Results:")
        print(f"- Total predictions: {len(predictions['predictions'])}")
        print("- Class distribution:")
        for cls, count in predictions['prediction_counts'].items():
            print(f"  {cls}: {count} ({count/len(predictions['predictions']):.1%})")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure:")
        print("1. Data files exist in emg_data/ directory")
        print("2. Files are properly formatted CSV with labels in last column")
        print(f"Current working directory: {os.getcwd()}")
