import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

DATA_PATH = r"C:/Users/shrih/OneDrive/Desktop/incxrdata/archive (8)/INCXR"

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def load_datasets(num_clients, cid):
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=get_transforms())

    # shuffle indices
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    # split among clients
    split = np.array_split(indices, num_clients)
    client_indices = split[cid]

    # train/val split (80/20)
    split_point = int(0.8 * len(client_indices))
    train_idx = client_indices[:split_point]
    val_idx = client_indices[split_point:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

    return train_loader, val_loader