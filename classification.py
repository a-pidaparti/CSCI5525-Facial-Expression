import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader

from Dataloader import FaceDataset
from ResNet9 import ResNet9

os.chdir('B:\CSCI_5525_Project/models/')
class ConvBlock(nn.Module):
    def __init__(self, channels, dim):
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(dim[0] * dim[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout(),
        )
    
    def forward(self, x):
        return self.layers(x)


class FaceClassifier(nn.Module):
    def __init__(self, channels, dim, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=576, out_features=576)
        self.fc2 = nn.Linear(in_features=576, out_features=7)

        self.layers = nn.Sequential(
            ConvBlock(channels, dim),
            ConvBlock(channels, dim / 2),
            ConvBlock(channels, dim / 4),
            ConvBlock(channels, dim / 8),
            
        )
    
    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.relu(tmp)
        tmp = self.pool(tmp)
        tmp = torch.flatten(tmp, start_dim=1)
        tmp = self.fc1(tmp)
        tmp = self.fc2(tmp)
        return F.log_softmax(tmp)

def train_one_epoch(model, train_loader, optimizer):
    for idx, data in enumerate(train_loader):
        print(idx, " / ", len(train_loader.dataset) / 64)
        X, y = data
        model.zero_grad()
        output = model(X.float())
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

    print(loss)
    torch.save(model.state_dict(), './classifier_weights.pth')
    return model

def eval(model, val_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        for jdx, data in enumerate(val_loader):
            X, y = data
            output = model(X.float())
            print(jdx, ' / ', len(val_loader.dataset) / 64)
            for idx, i in enumerate(output):
                if torch.argmax(i).item() == y[idx].item():
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))

def get_transform():
    return transforms.Compose([
    ])
    
    
def main():
    train_path = "B:\CSCI_5525_Project/images/train/"
    val_path = "B:\CSCI_5525_Project/images/validation/"

    # transform = get_transform()

    train_set = FaceDataset(train_path)
    val_set = FaceDataset(val_path)

    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=64, shuffle=False)

    model = ResNet9(1, 7)

    try:
        l = torch.load("./classifier_weights.pth")
        model.load_state_dict(l)
        print("loaded model")
    except:
        print("no model found")


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(10):
        model = train_one_epoch(model, train_loader, optimizer=optimizer)
        eval(model, val_loader)

if __name__ == "__main__":
    main()