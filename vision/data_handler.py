import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import config

def get_classes():
    dataset_train = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=None,
        target_transform=None
    )

    return dataset_train.classes

def build_dataloaders():
    data_transform = transforms.Compose(
        [
            # Resize to 64x64
            transforms.Resize(size=(config.IMAGE_SIZE, config.IMAGE_SIZE)),
            # Flip randomly
            transforms.RandomHorizontalFlip(p=0.5),
            # Turn into tensor
            transforms.ToTensor()
        ]
    )

    dataset_train = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=data_transform,
        target_transform=None
    )

    dataset_test = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=data_transform,
        target_transform=None
    )
    
    class_names = dataset_train.classes

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=config.BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=config.BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True
    )

    return dataloader_train, dataloader_test, class_names
