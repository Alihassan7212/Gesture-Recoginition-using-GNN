import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GestureDataset
from model import GestureGNN
from utils import train_model, evaluate_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset
dataset = GestureDataset("path/to/gesture/data")

# Split dataset into train and test sets
train_set, test_set = dataset.split_train_test(train_ratio=0.8)

# Initialize model
model = GestureGNN(input_size=dataset.feature_size, hidden_size=128, output_size=dataset.num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_set, criterion, optimizer, device)

# Evaluate the model
accuracy = evaluate_model(model, test_set, device)

print(f"Test Accuracy: {accuracy}")
