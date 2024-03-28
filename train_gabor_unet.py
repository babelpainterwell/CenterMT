import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data import CenterlineDataset
from gabor_unet_model import GaborUNet
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Add augmentation to the dataset


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    average_loss = train_loss / len(train_loader)
    return average_loss


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        # for data, target in test_loader:
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()
            pred = output > 0.5
            correct_pixels += pred.eq(target).sum().item()
            total_pixels += target.numel()
            if batch_idx % args.log_interval == 0:
                print(f'Test Epoch: {epoch} [{batch_idx * len(data)}/{len(test_loader.dataset)}'
                      f' ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')

    average_loss = test_loss / total_pixels
    accuracy = 100. * correct_pixels / total_pixels
    return average_loss, accuracy

def main():
    # COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(description='PyTorch GaborUNet Training')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()



    # LOAD DATASET
    img_dir = 'samples'
    label_dir = 'labels'

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.ToTensor(),  
    ])

    dataset = CenterlineDataset(img_dir, label_dir, transform=transform)
    
    # Determine lengths for train and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Dataset Preparation Complete! Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")


    # TRAINING
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = GaborUNet(kernel_size=7, in_channels=1, out_channels=1, num_orientations=8, num_scales=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print("Training Started!")

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(args, model, device, test_loader, epoch)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step()

    torch.save(model.state_dict(), 'gabor_unet_model_state_dict.pth')
    print("Model's state_dict saved to gabor_unet_model_state_dict.pth")

    torch.save(model, 'gabor_unet_model_complete.pth')
    print("Entire model saved to gabor_unet_model_complete.pth")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), test_accuracies, color='red', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(train_losses)
    print(test_losses)


    # model = GaborUNet(kernel_size=7, in_channels=1, out_channels=1, num_orientations=8, num_scales=5)
    # model.load_state_dict(torch.load('gabor_unet_model_state_dict.pth'))
    # model.eval() 

    # model = torch.load('gabor_unet_model_complete.pth')
    # model.eval()  # Set the model to inference mode

if __name__ == '__main__':
    main()
