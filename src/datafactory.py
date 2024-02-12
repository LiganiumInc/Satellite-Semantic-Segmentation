import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from patchify import patchify
import glob
import cv2



BGR_classes = {'Water' : [ 41, 169, 226],
                'Land' : [246,  41, 132],
                'Road' : [228, 193, 110],
                'Building' : [152,  16,  60], 
                'Vegetation' : [ 58, 221, 254],
                'Unlabeled' : [155, 155, 155]} # in BGR

bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

def preprocess_mask(mask):
    
    cls_mask = np.zeros(mask.shape)  
    cls_mask[mask == BGR_classes['Water']] = bin_classes.index('Water')
    cls_mask[mask == BGR_classes['Land']] = bin_classes.index('Land')
    cls_mask[mask == BGR_classes['Road']] = bin_classes.index('Road')
    cls_mask[mask == BGR_classes['Building']] = bin_classes.index('Building')
    cls_mask[mask == BGR_classes['Vegetation']] = bin_classes.index('Vegetation')
    cls_mask[mask == BGR_classes['Unlabeled']] = bin_classes.index('Unlabeled')
    cls_mask = cls_mask[:,:,0] 
    
    return cls_mask

class SatelliteDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.IMG_NAMES = sorted(glob.glob(self.root + '/*/images/*.jpg'))
        
    def __len__(self):
        return len(self.IMG_NAMES)
    
    def __getitem__(self, idx):
        img_path = self.IMG_NAMES[idx]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cls_mask = preprocess_mask(mask)

        return image, cls_mask


class SatellitePatchesDataset(Dataset):
    def __init__(self, patches_images, patches_masks, transform=None):
        
        self.patches_images = patches_images
        self.patches_masks = patches_masks
        self.transform = transform

    def __len__(self):
        return len(self.patches_images)

    def __getitem__(self, idx):
        patch_image = self.patches_images[idx] 
        patch_mask = self.patches_masks[idx]
        
        if self.transform:
            transformed = self.transform(image=patch_image, mask=patch_mask)
            patch_image = transformed["image"]
            patch_mask = transformed["mask"]
            

        return patch_image, patch_mask

    

def create_dataloaders( dataset, batch_size, num_workers, train = True):
  """
  
  """
  
  shuffle = True if train else False
  
  # Turn images into data loaders
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers,
      pin_memory=True,
  )
  
  return dataloader 


if __name__ == '__main__':
    # Example usage
    root = '../DubaiDataset/'
    patch_size = 256  # Choose your desired patch size

    transform = transforms.Compose([transforms.ToTensor()])

    # Create the base dataset
    base_dataset = SatelliteDataset(root=root)
    print(len(base_dataset))

    # Create the patch dataset
    patch_dataset = SatellitePatchesDataset(base_dataset, patch_size, transform=transform)
    print(len(patch_dataset))

    # # Use DataLoader with the patch dataset
    # dataloader = DataLoader(patch_dataset, batch_size=32, shuffle=True)

    # # Now you can iterate through the dataloader in your training loop
    # for patches_images, patches_masks in dataloader:
    #     # Training logic here
    #     pass
