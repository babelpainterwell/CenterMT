import torch 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image  
import os


img_dir = 'samples'
label_dir = 'labels'


class CenterlineDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img.endswith('.png')]
        self.labels = [label for label in os.listdir(label_dir) if label.endswith('.png')]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        image = Image.open(img_path)
        label = Image.open(label_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
    

# main 

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CenterlineDataset(img_dir, label_dir, transform=transform)
    
    # Determine the lengths for train and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Dataset Preparation Completes! Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

