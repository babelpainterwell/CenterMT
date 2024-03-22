import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from dataset_preparation import MyDataset, train_loader, test_loader
from gabor_unet_model import GaborUNet
import argparse

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Moves the data and target tensors to the specified computing device
        optimizer.zero_grad() # gradients accumulate by default after each .backward() call
        output = model(data) # shape: [batch_size, num_classes]
        loss = F.nll_loss(output, target) # binary 
        loss.backward()  # Removed retain_graph=True 
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0 # Initializes variables to keep track of the total loss and the number of correctly predicted samples
    correct = 0
    with torch.no_grad(): # Disables gradient computation, saving memory and computations since gradients are not needed for evaluation
        for data, target in test_loader: # target shape: [batch_size]
            data, target = data.to(device), target.to(device)
            output = model(data) # shape: [batch_size, num_classes]
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Accumulates the sum of the loss for each batch. The reduction='sum' argument makes sure the loss is summed over the batch.
            pred = output.argmax(dim=1, keepdim=True)  # shape: [batch_size, 1]
            correct += pred.eq(target.view_as(pred)).sum().item() # reshape target to have the same shape as pred, then compare the two tensors element-wise and sum the number of correct predictions

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = GaborUNet(kernel_size=7, in_channels=1, num_orientations=8, num_scales=5).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

if __name__ == '__main__':
    main()