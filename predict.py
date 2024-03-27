import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


model = torch.load('gabor_unet_model_complete.pth')
model.eval()  # Set the model to inference mode

# Define a transform to convert the image to tensor and normalize it if necessary
transform = transforms.Compose([
    transforms.Grayscale(),  
    transforms.ToTensor(),
])

def load_image(image_path):
    """Load and transform an image."""
    image = Image.open(image_path)
    image = image.crop((1, 1, 65, 65))
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def predict(model, image_path):
    """Run model prediction on an image and display original and segmented images."""
    image = load_image(image_path)
    with torch.no_grad():  
        output = model(image)
        # Convert to binary image, threshold at 0.5 for demonstration
        segmented = output.squeeze().numpy() > 0.5
    # Display the original and the segmented image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    original = Image.open(image_path)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(segmented, cmap='gray')
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')
    plt.show()

test_image_path = 'samples/sample_1.png'
predict(model, test_image_path)