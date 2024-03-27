import os
import torch
from torchvision import transforms
from PIL import Image, ImageOps
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

def save_image(image, save_path):
    """Save the PIL or Tensor image."""
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.squeeze().cpu())  # Convert tensor to PIL image
    image.save(save_path)

def create_overlay(raw_image, output):
    """Create and return an overlay of the detected output on the raw image."""
    output = output.squeeze().cpu().numpy()
    segmentation = (output > 0.5) * 255  # Convert to binary and scale to 0-255
    segmentation_image = Image.fromarray(segmentation.astype(np.uint8))
    segmentation_image = segmentation_image.convert("RGBA")
    overlay = ImageOps.colorize(segmentation_image, 'red', 'red')
    
    raw_image = raw_image.convert("RGBA")
    return Image.blend(raw_image, overlay, 0.5)

validation_dir = 'validation'
detection_dir = 'detection'
if not os.path.exists(detection_dir):
    os.makedirs(detection_dir)

# Predict, create overlay, and save each image in the validation directory
for i, image_name in enumerate(sorted(os.listdir(validation_dir))):
    if image_name.endswith('.png'):
        image_path = os.path.join(validation_dir, image_name)
        raw_image = Image.open(image_path).crop((1, 1, 65, 65))
        
        image_tensor = load_image(image_path)
        with torch.no_grad():
            output = model(image_tensor)
        
        # Generate and save the overlay
        overlay_image = create_overlay(raw_image, output)
        overlay_save_path = os.path.join(detection_dir, f'overlay_{i + 1}.png')
        overlay_image.save(overlay_save_path)

print("Overlay images have been generated and saved in the 'detection' directory.")