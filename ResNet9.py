import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def conv_block(in_channels, out_channels, pool=False, pool_size=2):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU()
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=pool_size))

    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(conv_block(in_channels=in_channels, out_channels=32))
        self.blocks.append(conv_block(in_channels=32, out_channels=64, pool=True, pool_size=2))
        res_block = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.blocks.append(res_block)

        self.blocks.append(conv_block(64, 128, pool=True, pool_size=2))
        self.blocks.append(conv_block(128, 256, pool=True, pool_size=2))
        res_block_2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.blocks.append(res_block_2)

        self.blocks.append(nn.MaxPool2d(kernel_size=5))
        self.blocks.append(nn.Flatten())
        self.blocks.append(nn.Linear(256, num_classes))


    def forward(self, xb):
        x = xb
        for i in range(len(self.blocks)):
            identity = x
            x = self.blocks[i](x)
            if i == 2 or i == 5:
                x = x + identity

        return x





def train(trainset, testset, valset, model):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 10
    loss_list = []
    loss_list_mean = []
    iter = 0
    print(len(trainset.dataset))
    for epoch in range(num_epochs):

        print('Epoch: {}'.format(epoch))

        loss_buff = []

        for i, (images, labels) in enumerate(trainset):

            # getting the images and labels from the training dataset
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # clear the gradients w.r.t parameters
            optimizer.zero_grad()

            # now we can call the CNN and operate over the images
            outputs = model(images)

            # loss calculation
            loss = criterion(outputs, labels)
            loss_buff = np.append(loss_buff, loss.item())

            # backward for getting the gradients w.r.t. parameters
            loss.backward()

            loss_list = np.append(loss_list, (loss_buff))

            # update the parameters
            optimizer.step()

            iter += 1

            if iter % 10 == 0:
                print('Iterations: {}'.format(iter))

            ## VALIDATION PART#############
            # if iter % 100 == 0:
            #
            #     # Accuracy
            #     correct = 0
            #     total = 0
            #
            #     for images, labels in testset:
            #         images = images.requires_grad_().to(device)
            #         labels = labels.to(device)
            #
            #         outputs = model(images)
            #
            #         # get the predictions from the maximum value
            #         _, predicted = torch.max(outputs.data, 1)
            #
            #         # how many labels I have, also mean the size of the valid
            #         total += labels.size(0)
            #
            #         correct += (predicted == labels).sum()
            #
            #     accuracy = 100 * correct / total
            #
            #     print('Iterations: {}. Loss: {}. Validation Accuracy: {}'.format(iter,
            #                                                                      loss.item(), accuracy))
            loss_list_mean = np.append(loss_list_mean, (loss.item()))
            ################################

    # visualize the loss
    plt.plot(loss_list)
    plt.plot(loss_list_mean)
    torch.save(model.state_dict(), 'model_weights.pth')
    ## TEST PART#############


def main():
    data_full = datasets.MNIST(
        root='data',
        train='true',
        transform=transforms.ToTensor(),
        download=True,
    )

    train_data, test_data, valid_data = torch.utils.data.random_split(data_full,
                                                                      [30000, 20000, 10000])

    trainset = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    testset = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    valset = torch.utils.data.DataLoader(valid_data, batch_size=100, shuffle=True)

    model = ResNet9(1, 10)
    try:
        model = ResNet9(1, 10)
        model.load_state_dict(torch.load('model_weights.pth'))
        print('loaded weights')
    except:
        print('training weights')
        train(trainset, testset, valset, model)


    # Accuracy
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for images, labels in testset:

        images = images.requires_grad_().to(device)
        labels = labels.to(device)
        outputs = model(images)

        # get the predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # how many labels I have, also mean the size of the valid
        total += labels.size(0)

        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    print('Test Accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    main()
