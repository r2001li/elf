import os
from pathlib import Path
import json

import torch
from torch import nn, rand

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from safetensors.torch import save_model

from vision import elfvision
from vision import vision_engine

IMAGE_SIZE = 64

MODEL_DIR = "vision/data/models"
MODEL_NAME = "model.safetensors"
CLASSES_NAME = "classes.json"

TRAIN_DIR = "vision/data/dataset/train"
TEST_DIR = "vision/data/dataset/test"

NUM_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32

def save_tensors(model: torch.nn.Module):
    model_path = get_model_path()
    print("Saving model...")
    save_model(model=model, filename=model_path)
    # torch.save(obj=model, f=model_save_path)
    print(f"Saved model to {model_path}")

def save_classes(classes):
    print("Saving class names...")

    dictionary = {}
    for i in range(0, len(classes)):
        index = str(i)
        dictionary[index] = classes[i]

    json_object = json.dumps(dictionary, indent=4)
    
    classes_path = get_classes_path()
    with open(classes_path, "w") as outfile:
        outfile.write(json_object)

    print(f"Saved class names to {classes_path}")

def model_exists():
    model_path = get_model_path()
    return os.path.isfile(model_path)

def classes_exists():
    classes_path = get_classes_path()
    return os.path.isfile(classes_path)

def get_model_path():
    model_dir = Path(MODEL_DIR) 
    model_path = model_dir / MODEL_NAME
    return model_path

def get_classes_path():
    classes_dir = Path(MODEL_DIR)
    classes_path = classes_dir / CLASSES_NAME
    return classes_path

def get_classes():
    dataset_train = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=None,
        target_transform=None
    )

    return dataset_train.classes

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def build_dataloaders():
    data_transform = transforms.Compose(
        [
            # Resize to 64x64
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            # Flip randomly
            transforms.RandomHorizontalFlip(p=0.5),
            # Turn into tensor
            transforms.ToTensor()
        ]
    )

    dataset_train = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=data_transform,
        target_transform=None
    )

    dataset_test = datasets.ImageFolder(
        root=TEST_DIR,
        transform=data_transform,
        target_transform=None
    )
    
    class_names = dataset_train.classes

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True
    )

    return dataloader_train, dataloader_test, class_names

def train_vision():
    # Stop if model file already exists
    if model_exists():
        print("Model already exists.")
        return
    
    print(f"Model not found. Training new model...")
    
    train_dataloader, test_dataloader, class_names = build_dataloaders()
    save_classes(class_names)

    # Instantiating a new neural network, loss function, and optimizer
    model = elfvision.ELFVisionNN(input_shape=3, output_shape=len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    # Train model
    device = get_device()
    vision_engine.train(model=model, 
                 train_dataloader=train_dataloader, 
                 test_dataloader=test_dataloader, 
                 loss_fn=loss_fn, 
                 optimizer=optimizer, 
                 epochs=NUM_EPOCHS, 
                 device=device)
    
    # Save model to file
    save_tensors(model=model)

