import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lightning as L

# Set seed
L.seed_everything(42, workers=True)

# Paths
CIFAR10_PATH = './data/CIFAR10'
MNIST_PATH = './data/MNIST'
FASHIONMNIST_PATH = './data/FashionMNIST'

# Global vars
BATCH_SIZE = 128
SUBSET_SIZE = 1000

MAX_EPOCHS = 100

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

def load_data(trainset, testset, batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE):

    # Create subset
    trainset_inds = torch.randperm(len(trainset))[:subset_size]
    trainset = torch.utils.data.Subset(trainset, trainset_inds)

    testset_inds = torch.randperm(len(testset))[:subset_size]
    testset = torch.utils.data.Subset(testset, testset_inds)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# RUN CIFAR10 EXPERIMENT
def CIFAR10_EXP(lightning_model):

    # Load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=True, transform=transform)

    # Create data loaders
    trainloader, testloader = load_data(trainset, testset)

    # Run the experiment
    trainer = L.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(lightning_model, trainloader, testloader)

# RUN MNIST EXPERIMENT
def MNIST_EXP(lightning_model):

    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download=True, transform=transform_gray)
    testset = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download=True, transform=transform_gray)

    # Create data loaders
    trainloader, testloader = load_data(trainset, testset)

    # Run the experiment
    trainer = L.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(lightning_model, trainloader, testloader)

# RUN FashionMNIST EXPERIMENT
def FashionMNIST_EXP(lightning_model):

    # Load FashionMNIST dataset
    trainset = torchvision.datasets.FashionMNIST(root=FASHIONMNIST_PATH, train=True, download=True, transform=transform_gray)
    testset = torchvision.datasets.FashionMNIST(root=FASHIONMNIST_PATH, train=False, download=True, transform=transform_gray)

    # Create data loaders
    trainloader, testloader = load_data(trainset, testset)

    # Run the experiment
    trainer = L.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(lightning_model, trainloader, testloader)