import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lightning as L

# Set seed
L.seed_everything(42, workers=True)

# Transformations to apply to the dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize images
])

transform_gray = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize images
])

# RUN CIFAR10 EXPERIMENT
def CIFAR10_EXP(lightning_model, batch_size=128):

    # Load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Run the experiment
    trainer = L.Trainer()
    trainer.fit(lightning_model, trainloader, testloader)

# RUN MNIST EXPERIMENT
def MNIST_EXP(lightning_model, batch_size=128):

    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Run the experiment
    trainer = L.Trainer()
    trainer.fit(lightning_model, trainloader, testloader)

# RUN FashionMNIST EXPERIMENT
def FashionMNIST_EXP(lightning_model, batch_size=128):

    # Load FashionMNIST dataset
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Run the experiment
    trainer = L.Trainer()
    trainer.fit(lightning_model, trainloader, testloader)