# Get MNIST Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

batch_size = 64
PATH = './MNIST.pth'

train_dataset = datasets.MNIST(root='../../Dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../../Dataset/', train=True, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 x 32 x 32 => 6 x 28 x 28
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 x 28 x 28 => 16 x 10 x 10

        self.pool = nn.MaxPool2d(2, 2)
        self.pd = nn.ZeroPad2d(2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pd(x)
        # x = tensor<batchsize, channels, x, y>
        # x = 64 * 1 * 32 * 32
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = 64 * 16 * 5 * 5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    lenet = LeNet5()
    train = False
    epoch = 10
    if train:
        criterion = nn.CrossEntropyLoss()
        optimization = optim.Adam(lenet.parameters(), lr=0.001)

        for i in range(epoch):
            running_loss = 0.0
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimization.zero_grad()
                outputs = lenet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimization.step()

                running_loss += loss.item()
                if j % 100 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, j + 1, running_loss / 100))
                    running_loss = 0.0

        print('Trainning Completed')
        torch.save(lenet.state_dict(), PATH)
    else:
        lenet.load_state_dict(torch.load(PATH))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = lenet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
