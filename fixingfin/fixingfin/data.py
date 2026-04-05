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

    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = np.array_split(indices, num_clients)
    client_indices = split[cid]

    # 🔥 70/15/15 split
    train_split = int(0.7 * len(client_indices))
    val_split = int(0.85 * len(client_indices))

    train_idx = client_indices[:train_split]
    val_idx = client_indices[train_split:val_split]
    test_idx = client_indices[val_split:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader