import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

PATH = './cifar_net.pth'
batch_size = 512

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.CIFAR10(root='../../Dataset/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='../../Dataset/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)


def convBNrelu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )


class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()
        self.layer = nn.Sequential(
            # a x a x in_channels => ((a - 1)/stride + 1) x ((a - 1)/stride + 1) x out_channels
            convBNrelu(in_channels, out_channels, 3, stride, padding=1),
            # ((a - 1)/stride + 1) x ((a - 1)/stride + 1) x out_channels
            # => ((a - 1)/stride + 1) x ((a - 1)/stride + 1) x out_channels
            # When stride = 1
            # a x a x in_channels => a x a x out_channels
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3)
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:  # F(x).shape != x.shape
            self.downsample = nn.Sequential(
                # a x a x in_channels => ((a - 1)/stride + 1) * ((a - 1)/stride + 1) * out_channels
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3)
            )

    def forward(self, x):
        return F.relu(self.layer(x) + self.downsample(x))  # x = F(x) + x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = convBNrelu(3, 64, 7, stride=2, padding=3)  # 224 x 224 x 3 => 112 x 112 x 64
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),  # 112 x 112 x 64 => 56 x 56 x 64
            residualBlock(64, 64),
            residualBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            residualBlock(64, 128),
            residualBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            residualBlock(128, 256),
            residualBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            residualBlock(256, 512),
            residualBlock(512, 512)
        )
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # 7 x 7 x 512
        x = self.avgpool(x)  # 1 x 1 x 512
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    train = True
    epoch = 10
    resnet = ResNet()
    if train:
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(resnet.parameters(), lr=0.01)

        for i in range(epoch):
            running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                images, labels = data
                opt.zero_grad()
                outputs = resnet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss
                if j % 5 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, j + 1, running_loss / 5))
                    running_loss = 0.0

        print('Trainning Completed')
        torch.save(resnet.state_dict(), PATH)
    else:
        resnet.load_state_dict(torch.load(PATH))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = resnet(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
