import torch
import torch.nn as nn
import torchmetrics

class unet_config:
    
    global_seed = 42
    
    NUM_CLASSES = 6                          # Number of classes in the dataset
    
    # Models
    IN_CHANNELS = 3
    LEVEL_CHANNELS = [64, 128, 256, 512]
    BOTTLENECK_CHANNEL = 1024
    UPSAMPLE = True
    UPSAMPLE_MODE = 'bilinear'
    MODEL_NAME = 'unet_model.pth'
    MODEL_SAVE_FOLDER = 'MODELS_REGISTER'
    
    # Training
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam
    LOGS_DIR = 'runs/unet_logs'
    METRICS = {'accuracy': torchmetrics.Accuracy(task="multiclass" ,num_classes=NUM_CLASSES),
               'dice': torchmetrics.Dice(num_classes=NUM_CLASSES),
               'JaccardIndex': torchmetrics.JaccardIndex(task="multiclass",num_classes=NUM_CLASSES)}

class config_attention_unet:
    ROOT = '../../DubaiDataset/'               # Folder where to store the dataset 
    NUM_CLASSES = 6                          # Number of classes in the dataset
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.2
    TEST_SIZE = 0.1
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    PATCH_SIZE = 256
    LEARNING_RATE = 0.001
    MODEL_NAME = 'attention_unet_model.pth'
    MODEL_SAVE_FOLDER = 'MODELS_REGISTER/'
    LOGS_DIR = 'runs/attention_unet_logs'







