import torch
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, data_path):
        # Load and preprocess your gesture data here
        # Store the features, labels, and any other relevant information
        
    def __len__(self):
        # Return the total number of samples in the dataset
        pass
    
    def __getitem__(self, idx):
        # Return the features and label for the given index
        pass
    
    def split_train_test(self, train_ratio):
        # Split the dataset into train and test sets based on the given ratio
        # Return two instances of GestureDataset, representing train set and test set
        pass
