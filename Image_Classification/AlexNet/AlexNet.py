import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

batch_size = 64
PATH = './cifar_net.pth'

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='../../Dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='../../Dataset/', train=False, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=0)  # 3 * 227 * 227 => 96 * 55 * 55
        self.mp1 = nn.MaxPool2d(3, stride=2, padding=0)  # 96 * 55 * 55 => 96 * 27 * 27
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)  # 96 * 27 * 27 => 256 * 27 * 27
        self.mp2 = nn.MaxPool2d(3, stride=2, padding=0)  # 256 * 27 * 27 => 256 * 13 * 13
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)  # 256 * 13 * 13 => 384 * 13 * 13
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)  # 384 * 13 * 13 => 384 * 13 * 13
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)  # 384 * 13 * 13 => 256 * 13 * 13
        self.mp3 = nn.MaxPool2d(3, stride=2, padding=0)  # 256 * 13 * 13 => 256 * 6 * 6
        self.do = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        # x => (3 * 227 * 227)
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp3(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = self.do(F.relu(self.fc1(x)))
        x = self.do(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    alexnet = AlexNet()
    train = True
    epoch = 10
    if train:
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(alexnet.parameters(), lr=0.001)

        for i in range(epoch):
            running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                images, labels = data
                opt.zero_grad()
                outputs = alexnet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss.item()
                if j % 2 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, j + 1, running_loss / 2))
                    running_loss = 0.0

        print('Trainning Completed')
        torch.save(alexnet.state_dict(), PATH)
    else:
        alexnet.load_state_dict(torch.load(PATH))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataset:
                images, labels = data
                outputs = alexnet(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
