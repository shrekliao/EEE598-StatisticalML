# -*- coding: utf-8 -*-
"""hw5_b_singhide

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CabpnRvDoKU1UyfX_BlAXIP_jLlnrqBy
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# functions to show an image
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""# (b) 1 fully connected hidden layer:"""

class singleHidNet(nn.Module):

    def __init__(self):
        super(singleHidNet, self).__init__()

        M = 100 #200

        self.fc1 = nn.Linear(3072, M) #32 * 32 * 3= 3072
        self.fc2 = nn.Linear(M, 10)

    def forward(self, x):
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = singleHidNet()

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

epochLim = 25

testAcc = np.zeros(epochLim)
trainAcc = np.zeros(epochLim)
# train the network
for epoch in range(epochLim):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d %%' % (
            100 * correct / total))
    testAcc[epoch] = 100 * correct / total

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train accuracy: %d %%' % (
            100 * correct / total))

    trainAcc[epoch] = 100 * correct / total

print('Finished Training')

plt.figure(1)
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration Number (Epoch)')
plt.legend(('Training', 'Testing'))
plt.show()