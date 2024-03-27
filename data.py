from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import functional as TF
import random

class CenterlineDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = sorted([img for img in os.listdir(img_dir) if img.endswith('.png')])
        self.labels = sorted([label for label in os.listdir(label_dir) if label.endswith('.png')])

        assert len(self.images) == len(self.labels), "The number of images and labels do not match!"

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        try:
            image = Image.open(img_path).convert("L")  # Grayscale conversion for images
            label = Image.open(label_path).convert("L")  # Grayscale conversion for labels

            # Crop the first row and column off, adjusting the size to 64x64
            # Assuming the original size is 65x65, crop to get (1, 1, 65, 65)
            image = image.crop((1, 1, 65, 65))
            label = label.crop((1, 1, 65, 65))

        except Exception as e:
            print(f"Error opening image or label at index {index}: {e}")
            return None, None

        
        # Always apply basic transformations if provided
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        # Additionally, apply random augmentations
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270]) 
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
        

        return image, label


    

