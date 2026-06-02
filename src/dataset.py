import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarlaSafetyDataset(Dataset):
    def __init__(self, metadata_csv, img_dir, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.img_dir = img_dir
        
        if transform is not None:
            self.transform = transform
        else:
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
        # 1. Read the frame number and format it as a 6-digit padded string with a .png extension
        frame_id = str(self.metadata.iloc[idx]['frame']).zfill(6)
        img_filename = f"{frame_id}.png"
        
        # 2. Update path to match the extracted 'validation/rgb-front/' directory structure
        img_name = os.path.join(self.img_dir, "validation", "rgb-front", img_filename)
        
        # Fallback check to ensure your script handles non-nested paths gracefully
        if not os.path.exists(img_name):
            img_name = os.path.join(self.img_dir, "rgb-front", img_filename)
            
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"[CRITICAL] Image index asset not found at destination path: {img_name}")
            
        image = Image.open(img_name).convert('RGB')
        
        # 3. Map values using the verified database column headers
        labels = {
            'traffic_light': torch.tensor(self.metadata.iloc[idx]['has_traffic_light'], dtype=torch.float32),
            'pedestrian': torch.tensor(self.metadata.iloc[idx]['has_pedestrian'], dtype=torch.float32),
            'vehicle': torch.tensor(self.metadata.iloc[idx]['has_vehicle'], dtype=torch.float32)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels