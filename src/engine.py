
import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import  MulticlassAccuracy, Dice, MulticlassJaccardIndex
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def train_step(model: Module, data: tuple, optimizer: Optimizer, criterion, metrics: dict, device: torch.device) -> float:
    """
    Performs one training step.

    Args:
        model (torch.nn.Module): The neural network model.
        data (tuple): A tuple containing input data and ground truth labels.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        criterion: The loss criterion.
        metrics (dict): Dictionary containing torchmetrics metrics for training.
        device (torch.device): The device on which the data and model should be placed.

    Returns:
        
    """
    model.train()
    inputs, targets = data
    # print("targets : ", targets, targets.shape)
    inputs, targets = inputs.to(device), torch.tensor(targets, dtype=torch.long, device=device)
    
    optimizer.zero_grad()

    outputs = model(inputs)
    
    # Calculate loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    loss.backward()
    
    optimizer.step()

    # Update and log metrics
    probs = torch.softmax(outputs, dim=1)
    masks_pred = torch.argmax(probs, dim=1)

    for metric_name, metric in metrics.items():
        metric.to(device).update(masks_pred, targets) # targets.to(torch.long) is added to have integer type for targets as it is for the preds
        # wandb.log({f'Train/{metric_name}': metric.compute()})

    return loss.item(), metrics 


def val_step(model: Module, data: tuple, criterion, metrics: dict, device: torch.device) -> float:
    """
    Performs one validation step.

    Args:
        model (torch.nn.Module): The neural network model.
        data (tuple): A tuple containing input data and ground truth labels.
        criterion: The loss criterion.
        metrics (dict): Dictionary containing torchmetrics metrics for validation.
        device (torch.device): The device on which the data and model should be placed.

    Returns:
        float: Validation loss.
    """
    model.eval()
    inputs, targets = data
    inputs, targets = inputs.to(device), torch.tensor(targets, dtype=torch.long, device=device)

    outputs = model(inputs)
    
    # Calculate loss
    loss = criterion(outputs, targets)

    # Update and log metrics
    probs = torch.softmax(outputs, dim=1)
    masks_pred = torch.argmax(probs, dim=1)
    for metric_name, metric in metrics.items():
        metric.to(device).update(masks_pred, targets)
        # wandb.log({f'Validation/{metric_name}': metric.compute()})

    return loss.item(), metrics


def train(model: Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: Optimizer, criterion, 
          metrics: dict,
          num_classes : int, num_epochs: int, 
          device: torch.device, tensorboard_writer ) -> None:
    """
    Trains the neural network model using a generic loss criterion and logs metrics to Weights & Biases.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        criterion: The loss criterion.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        device (str or torch.device, optional): The device on which the model should be trained. Defaults to 'cuda'.

    Returns:
        None
    """
    # wandb.init(project="Satellite Image Semantic Segmentation", name="training_run")

    ############## Initialize torchmetrics metrics ################################################
    train_metrics = metrics
    
    val_metrics = metrics

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        
        ########################### Training phase ################################################
        model.train()
        train_loss = 0.0

        for data in train_loader:
            loss_batch, train_metrics = train_step(model, data, optimizer, criterion, train_metrics, device)
            train_loss += loss_batch

        # compute loss  over all training batches and log  to tensorboard
        avg_train_loss = train_loss / len(train_loader)
        tensorboard_writer.add_scalar('Loss/train', avg_train_loss, epoch)  # log training loss to tensorboard
        
        # compute metrics   over all training batches and log  them to tensorboard
        for metric_name, metric in train_metrics.items():
            tensorboard_writer.add_scalar(f'Train/{metric_name}', metric.compute(), epoch)
            
        
        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                loss_batch, val_metrics = val_step(model, data, criterion, val_metrics, device)
                val_loss += loss_batch

        # compute loss  over all validation batches and log  to tensorboard
        avg_val_loss = val_loss / len(val_loader)
        tensorboard_writer.add_scalar('Loss/val', avg_val_loss, epoch)  # log validation loss to tensorboard
        
        # compute metrics   over all validation batches and log  them to tensorboard
        for metric_name, metric in val_metrics.items():
            tensorboard_writer.add_scalar(f'Val/{metric_name}', metric.compute(), epoch)
            

        # Display progress of loss and metrics
        print(f'\nEpoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}')
        print(f'Train Dice Coeff.: {train_metrics["dice"].compute():.4f} - Val Dice Coeff.: {val_metrics["dice"].compute():.4f}')

        # Reset metric states after each epoch
        for metric_name, metric in val_metrics.items():
            metric.reset()
            metric.reset()
            
    # wandb.finish()


# import torch
# from torchmetrics import ConfusionMatrix, Dice, Jaccard
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix

def predict(model, dataloader, num_classes, device='cuda'):
    """
    Performs predictions using a pre-saved model on a dataloader and computes various metrics.

    Args:
        model (torch.nn.Module): The pre-saved neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for prediction data.
        num_classes (int): Number of classes in the classification task.
        device (str or torch.device, optional): The device on which the model should be run. Defaults to 'cuda'.

    Returns:
        list: List containing computed metrics in the order [MulticlassAccuracy, Dice, MulticlassJaccardIndex].
    """
    model.eval()
    model.to(device)

    multiclass_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    dice_metric = Dice(num_classes=num_classes).to(device)
    jaccard_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Predicting'):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())


    # Compute confusion matrix
    confusion_mat = confmat(all_preds, all_targets)

    # Print and return metrics
    metrics_list = [multiclass_accuracy(all_preds, all_targets),
                    dice_metric(all_preds, all_targets),
                    jaccard_metric(all_preds, all_targets)]

    print(f"Multiclass Accuracy: {metrics_list[0]:.4f}")
    print(f"Dice: {metrics_list[1]:.4f}")
    print(f"Jaccard Index: {metrics_list[2]:.4f}")
    
    print("Confusion Matrix:")
    confusion_mat.plot(normalized=True)

    return metrics_list


