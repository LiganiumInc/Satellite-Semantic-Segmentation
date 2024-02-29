import torch
import torch.nn as nn
import torchmetrics



class attention_unet_config:
    
    global_seed = 42
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    NUM_CLASSES = 6                          # Number of classes in the dataset
    
    # Models
    IN_CHANNELS = 3
    LEVEL_CHANNELS = [64, 128, 256]
    BOTTLENECK_CHANNEL = 512
    MODEL_NAME = 'attention_unet_model.pth'
    MODEL_SAVE_FOLDER = 'MODELS_REGISTER'
    
    # Training
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    class_weights = [ 1.27769539,  0.31187535,  1.7162232 ,  1.26614547,  1.64332638, 33.40421679]
    LOSS_FN = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float,device = DEVICE),reduction='mean')
    OPTIMIZER = torch.optim.Adam
    LOGS_DIR = 'runs/attention_unet_logs'
    METRICS = {'accuracy': torchmetrics.Accuracy(task="multiclass" ,num_classes=NUM_CLASSES),
               'dice': torchmetrics.Dice(num_classes=NUM_CLASSES),
               'JaccardIndex': torchmetrics.JaccardIndex(task="multiclass",num_classes=NUM_CLASSES)}









