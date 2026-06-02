import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarlaSafetyDataset(Dataset):
    """
    Custom interface to parse and preprocess camera frames collected from the
    CARLA simulator under clear weather and daytime conditions.
    """
    def __init__(self, metadata_csv, img_dir, transform=None):
        """
        Args:
            metadata_csv (str): Path to the log mapping filenames to binary target metrics.
            img_dir (str): Base target directory housing raw image assets.
            transform (callable, optional): Standard image transformation sequences.
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.img_dir = img_dir
        
        if transform is not None:
            self.transform = transform
        else:
            # Implement empirical ImageNet standard normalization parameters
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['image_file'])
        image = Image.open(img_name).convert('RGB')
        
        # Extract binary labels for the targeted classification tasks
        labels = {
            'traffic_light': torch.tensor(self.metadata.iloc[idx]['traffic_light'], dtype=torch.float32),
            'pedestrian': torch.tensor(self.metadata.iloc[idx]['pedestrian'], dtype=torch.float32),
            'vehicle': torch.tensor(self.metadata.iloc[idx]['vehicle'], dtype=torch.float32)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels