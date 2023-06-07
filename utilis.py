import torch

def train_model(model, train_set, criterion, optimizer, device):
    # Set model to training mode
    model.train()
    
    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    
    # Training loop
    for batch_features, batch_labels in train_loader:
        # Move data to the device
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_features)
        
        # Compute loss
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()

def evaluate_model(model, test_set, device):
    # Set model to evaluation mode
    model.eval()
    
    # Prepare data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    
    # Evaluation loop
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            # Move data to the device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counts
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
