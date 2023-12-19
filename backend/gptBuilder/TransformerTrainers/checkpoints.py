import os
import torch

checkpoint_path = "../../model_checkpoint.pth"
def save_checkpoint(model, optimizer, iteration, filename=checkpoint_path):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

# Function to load a checkpoint
def load_checkpoint(model, optimizer, filename=checkpoint_path):
    if os.path.isfile(filename):
        print("Loading checkpoint...")
        checkpoint = torch.load(filename)
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return iteration
    else:
        print("No checkpoint found, starting from scratch.")
        return 0