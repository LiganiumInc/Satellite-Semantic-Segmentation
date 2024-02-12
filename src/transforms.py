



import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


# Augment train data
train_transforms = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(max_value=255),  # Divide pixel values by max_value to get a float32 output array where all values lie in the range [0, 1.0]
        ToTensorV2(), # Convert image and mask to torch.Tensor. The numpy HWC image is converted to pytorch CHW tensor.
    ]
)

# Don't augment test data, only reshape
val_transforms = A.Compose(
    [
        A.ToFloat(max_value=255),
        ToTensorV2()
    ]
)