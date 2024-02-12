import torch
import os
import numpy as np
from matplotlib import pyplot as plt
import copy
from patchify import patchify
import glob
from sklearn.model_selection import train_test_split 
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

# seeding function for reproducibility

def seed_everything(seed : int=42):
    
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    

def generate_patches(base_dataset, patch_size):
    
    patches_images = []
    patches_masks = []

    for idx in range(len(base_dataset)):
        image, mask = base_dataset[idx]

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Apply patchify to image and mask
        patches_image = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)

        # Reshape patches
        patches_image = patches_image.reshape((-1, patch_size, patch_size, 3))
        patches_mask = patches_mask.reshape((-1, patch_size, patch_size))

        patches_images.extend(patches_image)
        patches_masks.extend(patches_mask)

    return np.array(patches_images), np.array(patches_masks)

def split_train_val_test(X, y, test_size=0.2, val_size=0.25, random_state=None):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - X: Features
    - y: Labels
    - test_size: The proportion of the dataset to include in the test split
    - val_size: The proportion of the dataset to include in the validation split
    - random_state: Seed for random number generation

    Returns:
    - X_train, X_val, X_test: Features for training, validation, and test sets
    - y_train, y_val, y_test: Labels for training, validation, and test sets
    
    Example usage:
        Assuming you have X and y as your feature and label arrays
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.2, val_size=0.25, random_state=42)
    
    """
    
    # Split the dataset into train and temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)

    # Split the temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

   

def display_random_image_and_mask(images, masks, tensorboard_writer=None):
    """
    Display a random image and its corresponding mask side by side.

    Parameters:
    - images: List or array of images
    - masks: List or array of corresponding masks
    
    # Example usage:
         Assuming you have 'patches_images' and 'patches_masks' as your image and mask arrays
         display_random_image_and_mask(patches_images, patches_masks)

    """
    random_image_id = random.randint(0, len(images) - 1)
    random_image = images[random_image_id]
    random_mask = masks[random_image_id]

    plt.figure(figsize=(10, 5))

    # Display the random image
    plt.subplot(121)
    plt.imshow(random_image)
    plt.title("Random Image")

    # Display the corresponding mask
    plt.subplot(122)
    plt.imshow(random_mask)
    plt.title("Corresponding Mask")

    plt.show()
    
    # write to tensorboard
    if tensorboard_writer:
        image_tensor = torch.tensor(random_image).permute(2, 1, 0)
        mask_tensor = torch.tensor(random_mask) * 255
        tensorboard_writer.add_image('Random Image', image_tensor)
        tensorboard_writer.add_image('Corresponding Mask', mask_tensor.unsqueeze(0))
        # tensorboard_writer.add_image("Random Image and Mask", torch.cat([image_tensor, mask_tensor.unsqueeze(0)], dim=0))
    

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTime on {device}: {total_time:.3f} seconds")

    return round(total_time,3)

def visualize_augmentations(dataset, idx=0, samples=5):
    
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    

def load_model(model_save_path, model_name):
    """
    Loads a PyTorch model from a specified path.

    Args:
        model_save_path (str): The path to the saved model file.
        model_name (str): The name of the model class.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    possible_extensions = ['.pth', '.pt']
    
    for ext in possible_extensions:
        model_path = os.path.join(model_save_path, f"{model_name}{ext}")
        
        if os.path.exists(model_path):
            try:
                # Load the model
                model = torch.load(model_path)
                print(f"Model '{model_name}' loaded successfully from '{model_path}'.")
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading model from '{model_path}': {e}")

    raise FileNotFoundError(f"Model file not found at paths: {[os.path.join(model_save_path, f'{model_name}{ext}') for ext in possible_extensions]}")

# Example usage:
# loaded_model = load_model(model_save_path='/path/to/models', model_name='your_model_name')
