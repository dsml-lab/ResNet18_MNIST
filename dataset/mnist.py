from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_train_dataloader(root: str, batch_size: int=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=90)
    ])

    train_dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_dataloader

def get_test_dataloader(root: str, batch_size: int=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=90)
    ])

    test_dataset = MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    test_dataloader= DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    return test_dataloader