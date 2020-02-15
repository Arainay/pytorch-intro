import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43, 0.44, 0.47], std=[0.20, 0.20, 0.20])
])


def get_train_set():
    train_set = datasets.SVHN(
        './data/',
        split='train',
        transform=transform
    )

    return train_set


def get_test_set():
    test_set = datasets.SVHN(
        './data/',
        split='test',
        transform=transform
    )

    return test_set


def get_train_and_val_loaders(train_data, batch_size=64):
    data_size = train_data.data.shape[0]
    validation_split = 0.2
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=val_sampler
    )

    return train_loader, val_loader


def compute_accuracy(model, loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
