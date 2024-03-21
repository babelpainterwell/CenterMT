import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# FOR COORDINATE-BASED CENTERLINE DETECTION



class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None):
        super(MyDataset, self).__init__()
        self.imgs = []
        self.labels = []
        self.transform = transform

        with open(os.path.join(root, datatxt), 'r') as file:
            for line in file:
                line = line.rstrip()
                words = line.split()
                self.imgs.append(os.path.join(root, words[0]))
                self.labels.append(int(words[1]))

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.imgs)

def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

temp_dataset = MyDataset(root='./', datatxt='train.txt', transform=transforms.ToTensor())

# Calculate mean and std without normalization
mean, std = calculate_mean_std(temp_dataset)
print(f'Mean: {mean}, Std: {std}')

# Now, use these calculated values in actual dataset transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = MyDataset(root='./', datatxt='train.txt', transform=transformations)
test_dataset = MyDataset(root='./', datatxt='test.txt', transform=transformations)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=1)


