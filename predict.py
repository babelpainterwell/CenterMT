import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = torch.load('gabor_unet_model_complete.pth')
model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_image(image_path):
    """Load and transform an image."""
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = image.crop((1,1,65,65)) 
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image.to(device)

def save_segmentation(output, save_path):
    """Save the segmentation output."""
    output = output.squeeze().cpu().numpy()
    output = (output > 0.5) * 255  # Convert to binary and scale to 0-255
    img = Image.fromarray(output.astype(np.uint8))
    img.save(save_path)

# def save_raw(raw, save_path):
#     raw = raw.squeeze().cpu().numpy()
#     img = Image.fromarray(raw.astype(np.uint))

validation_dir = 'validation'
detection_dir = 'detection'
if not os.path.exists(detection_dir):
    os.makedirs(detection_dir)

# Predict and save each image in the validation directory
for i, image_name in enumerate(os.listdir(validation_dir)):
    if image_name.endswith('.png'):
        image_path = os.path.join(validation_dir, image_name)
        image = load_image(image_path)
        
        with torch.no_grad():
            output = model(image)
        
        folder_dir = os.path.join(detection_dir, f'detection_{i}')

        output_save_path = os.path.join(folder_dir, f'output_{i}')
        raw_save_path = os.path.join(folder_dir, f'raw_{i}')
        save_segmentation(output, output_save_path)

print("Prediction completed and saved in 'detection' directory.")
