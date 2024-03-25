from torch.utils.data import Dataset
from PIL import Image
import os


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
        except Exception as e:
            print(f"Error opening image or label at index {index}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

    

