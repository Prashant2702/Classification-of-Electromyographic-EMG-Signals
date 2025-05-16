import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import multiprocessing
import openai

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load numerical dataset
def load_numerical_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Train numerical model with parallelization
def train_numerical_model(filepath):
    X, y = load_numerical_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Memory-optimized RandomForest with fewer estimators and warm start
    model = RandomForestClassifier(n_estimators=10,  # Start with 10 estimators
                                random_state=42,
                                n_jobs=-1,
                                warm_start=True)  # Enable incremental training
    
    # Train in smaller chunks to reduce memory usage
    for i in range(4):  # 4 chunks of 10 estimators = 40 total
        model.n_estimators += 10
        model.fit(X_train, y_train)
        print(f"Trained {model.n_estimators} estimators")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    return model

# Optimized CNN model with fewer parameters & mixed precision support
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Reduced channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Reduced channels
        self.fc1 = nn.Linear(32 * 32 * 32, 64)  # Smaller FC layer
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Optimized DataLoader with multi-threading
def load_image_data(image_folder, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Reduced resolution
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=image_folder, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True)
    
    return train_loader, val_loader

# Train CNN model with mixed precision
def train_image_model(image_folder, epochs=10, lr=0.001):
    train_loader, val_loader = load_image_data(image_folder)
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    return model

# File path for numerical EMG data
NUMERICAL_EMG_DATA_PATH = "EMG-data.csv"

def load_trained_model():
    """Load pre-trained model from file"""
    import joblib
    try:
        model = joblib.load('trained_model.pkl')
        print("Loaded pre-trained model")
        return model, True
    except FileNotFoundError:
        print("No pre-trained model found")
        return None, False

def save_model(model):
    """Save trained model to file"""
    import joblib
    joblib.dump(model, 'trained_model.pkl', compress=3)  # High compression to reduce memory
    print("Model saved successfully")

def predict_emg_data(model, data):
    """Make predictions on EMG data"""
    try:
        # Ensure data is in correct format
        if isinstance(data, pd.DataFrame):
            X = data.values
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError("Input data must be pandas DataFrame or numpy array")
            
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

# Only train if explicitly called
if __name__ == '__main__':
    model = train_numerical_model(NUMERICAL_EMG_DATA_PATH)
    save_model(model)
