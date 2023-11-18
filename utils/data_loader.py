# -*- coding: utf-8 -*-

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Data_Loader():
    """
    Example
    -------
        >>> train_loader, test_loader = Data_Loader("mnist", "./data/", 64).load_data()
    """

    def __init__(self, dataset_name, data_path, train_batch_size, test_batch_size=1000):
        """
        Parameters
        ----------
            dataset_name : str, options={"mnist", "fashionmnist", "cifar10", "cifar100"}
            
            data_path : str

            train_batch_size : int

            test_batch_size : int, default=1000
        """

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
    
    def load_data(self):
        if self.dataset_name == 'mnist':
            return load_mnist(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'fashionmnist':
            return load_fashionmnist(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'cifar10':
            return load_cifar10(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'cifar100':
            return load_cifar100(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == "tiny-imagenet":
            return load_tiny_imagenet(self.data_path, self.train_batch_size, self.test_batch_size)


def load_mnist(data_path, train_batch_size, test_batch_size):
    train_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_fashionmnist(data_path, train_batch_size, test_batch_size):
    train_data = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_data = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_cifar10(data_path, train_batch_size, test_batch_size):
    train_data = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    )
    test_data = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_cifar100(data_path, train_batch_size, test_batch_size):
    train_data = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    test_data = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def load_tiny_imagenet(data_path, train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                            std=[0.2770, 0.2691, 0.2821])])

    train_data = datasets.ImageFolder(root=data_path + "train", transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4)

    test_data = datasets.ImageFolder(root=data_path + "val", transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader
