import torch
from torch import mps
from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAccuracy
from Models.single_layer_nn import SingleLayerTest
from loss import CrossEntropyLoss
from optimizer import HebbianDescent
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import wandb
import os
import gc


def train_epoch():
    epoch_loss = 0
    epoch_acc = 0

    for step, (x, y) in enumerate(train_loader):
        x = x.to(device=device)
        y = y.to(device=device)

        x = x.view(x.size(0), 28*28)
        h = model(x)
        optimizer.zero_grad()

        loss = loss_function(h, y)
        loss.backward(retain_graph=True)

        leaf_grad = grad(loss, h, create_graph=True, retain_graph=True)[0]
        # Derivative of the loss wrt output i.e DL/dh

        optimizer.step(x, leaf_grad)
        epoch_loss += loss.item()

        # Metrics
        y = y.to(device='cpu')
        h = h.to(device='cpu')
        epoch_acc += accuracy(h, y).item()

        del x, y, h
        mps.empty_cache()
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1)


def test_epoch():
    epoch_loss = 0
    epoch_acc = 0

    for step, (x, y) in enumerate(test_loader):
        x = x.to(device=device)
        y = y.to(device=device)

        x = x.view(x.size(0), 28*28)
        h = model(x)

        loss = loss_function(h, y)
        epoch_loss += loss.item()

        # Metrics
        y = y.to(device='cpu')
        h = h.to(device='cpu')
        epoch_acc += accuracy(h, y).item()

        del x, y, h
        mps.empty_cache()
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1)


def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)

        train_loss, train_acc = train_epoch()
        mps.empty_cache()
        model.eval()

        with torch.no_grad():
            test_loss, test_acc = test_epoch()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Test Loss: ", test_loss)
            print("Test Accuracy: ", test_acc)

            wandb.log({
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc
            })


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    load_dotenv('.env')

    mnist_train_dataset = MNIST(root=os.getenv("dataset"), train=True, transform=T.Compose(
        [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True)
    mnist_test_dataset = MNIST(root=os.getenv("dataset"), train=False, transform=T.Compose(
        [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]), download=True)

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    wandb.init(
        project="Hebbian Descent",
        config={
            "Architecture": "NN",
            "Dataset": "MNIST",
        }
    )
    train_loader = DataLoader(mnist_train_dataset, **params)
    test_loader = DataLoader(mnist_test_dataset, **params)

    device = torch.device("mps")
    model = SingleLayerTest(input_features=28*28).to(device=device)

    LR = 0.05
    NUM_EPOCHS = 100

    optimizer = HebbianDescent(model.parameters(), lr=LR)
    loss_function = CrossEntropyLoss()

    accuracy = MulticlassAccuracy(num_classes=10)

    training_loop()
