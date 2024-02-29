import torch
import torchmetrics
from torchmetrics.classification import  MulticlassAccuracy, Dice, MulticlassJaccardIndex, MulticlassConfusionMatrix
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def predict(model, dataloader, num_classes, tensorboard_writer=None, device='cuda'):
    """
    Performs predictions using a pre-saved model on a dataloader and computes various metrics.

    Args:
        model (torch.nn.Module): The pre-saved neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for prediction data.
        num_classes (int): Number of classes in the classification task.
        tensorboard_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard SummaryWriter. Defaults to None.
        device (str or torch.device, optional): The device on which the model should be run. Defaults to 'cuda'.

    Returns:
        list: List containing computed metrics in the order [MulticlassAccuracy, Dice, MulticlassJaccardIndex].
    """
    model.eval()
    model.to(device)

    multiclass_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    dice_metric = Dice(num_classes=num_classes).to(device)
    jaccard_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Predicting'):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            masks_pred = torch.argmax(probs, dim=1)
    
            all_preds.extend(masks_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    
    all_preds, all_targets = torch.tensor(all_preds).to(device), torch.tensor(all_targets, dtype=torch.long, device=device)
   
    # Compute confusion matrix
    confmat.update(all_preds, all_targets)

    # Print and return metrics
    metrics_list = [multiclass_accuracy(all_preds, all_targets),
                    dice_metric(all_preds, all_targets),
                    jaccard_metric(all_preds, all_targets)]

  
    print(f"Multiclass Accuracy: {metrics_list[0]:.4f}")
    multiclass_accuracy.plot()

    print(f"Dice: {metrics_list[1]:.4f}")
    dice_metric.plot()
    
    print(f"Jaccard Index: {metrics_list[2]:.4f}")
    jaccard_metric.plot()

    print("Confusion Matrix:")
    confmat.plot()

    if tensorboard_writer is not None:
        # Add confusion matrix to TensorBoard
        tensorboard_writer.add_image('Confusion Matrix', confmat.compute().numpy(), global_step=0)

    return metrics_list

# Example usage with TensorBoard
# tensorboard_writer = SummaryWriter('/path/to/tensorboard/logs')
# metrics = predict(model, test_loader, num_classes=6, tensorboard_writer=tensorboard_writer)