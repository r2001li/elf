import os

import torch
from torch import nn

from safetensors.torch import save_model

from pathlib import Path

from config import config
from vision import data_handler
from vision import model_builder
from vision import engine

def save_tensors(model: torch.nn.Module):
    print("Saving model")

    model_dir = Path(config.MODEL_DIR)
    model_path = model_dir / config.MODEL_NAME

    print(f"[INFO] Saving model to: {model_path}")
    save_model(model=model, filename=model_path)
    # torch.save(obj=model, f=model_save_path)

def main():
    train_dataloader, test_dataloader, class_names = data_handler.build_dataloaders()

    target_dir_path = Path(config.MODEL_DIR)
    model_path = target_dir_path / config.MODEL_NAME

    # Stop if model file already exists
    if os.path.isfile(model_path):
        print("Model file already found")
        return
    
    print(f"Model file not found at: {target_dir_path} Training new model")
    
    # Instantiating a new neural network, loss function, and optimizer
    model = model_builder.ELFVision2Model(input_shape=config.INPUT_SHAPE, output_shape=len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.LEARNING_RATE)

    # Train model
    print("Begin training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine.train(model=model, 
                 train_dataloader=train_dataloader, 
                 test_dataloader=test_dataloader, 
                 loss_fn=loss_fn, 
                 optimizer=optimizer, 
                 epochs=config.NUM_EPOCHS, 
                 device=device)
    
    # Save model to file
    save_tensors(model=model)

if __name__ == '__main__':
    main()