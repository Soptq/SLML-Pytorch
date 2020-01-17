import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

batch_size = 64
PATH = './cifar_net.pth'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

trainset = datasets.CIFAR10(root='../../Dataset', train=True, transform=transform, download=True)
testset = datasets.CIFAR10(root='../../Dataset', train=False, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)


def convBNrelu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )


class InceptionV1(nn.Module):
    def __init__(self, in_channel, conv1x1, conv3x3R, conv3x3, conv5x5R, conv5x5, pool_proj):
        super(InceptionV1, self).__init__()
        self.branch1x1 = convBNrelu(in_channel, conv1x1, 1)  # 28 x 28 x 192 => 28 x 28 x conv1x1
        self.branch3x3 = nn.Sequential(
            convBNrelu(in_channel, conv3x3R, 1),  # 28 x 28 x 192 => 28 x 28 x conv3x3R
            convBNrelu(conv3x3R, conv3x3, 3, padding=1)  # 28 x 28 x conv3x3R => 28 x 28 x conv3x3
        )
        self.branch5x5 = nn.Sequential(
            convBNrelu(in_channel, conv5x5R, 1),  # 28 x 28 x 192 => 28 x 28 x conv5x5R
            convBNrelu(conv5x5R, conv5x5, 5, padding=2)  # 28 x 28 x conv5x5R => 28 x 28 x conv5xt
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),  # 28 x 28 x 192 => 28 x 28 x 192
            convBNrelu(in_channel, pool_proj, 1)  # 28 x 28 x 192 => 28 x 28 x pool_proj
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)

        layer = torch.cat((b1, b2, b3, b4), dim=1)
        return layer


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.layer1 = nn.Sequential(
            convBNrelu(3, 64, 7, stride=2, padding=3),  # 224 x 224 x 3 => 112 x 112 x 64
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 112 x 112 x 64 => 56 x 56 x 64
            # nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            convBNrelu(64, 192, 3, stride=1, padding=1),  # 56 x 56 x 64 => 56 x 56 x 192
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 56 x 56 x 192 => 28 x 28 x 192
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            InceptionV1(192, 64, 96, 128, 16, 32, 32),  # 28 x 28 x 192 => 28 x 28 x 256
            InceptionV1(256, 128, 128, 192, 32, 96, 64),  # 28 x 28 x 256 => 28 x 28 x 480
            nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 28 x 28 x 480 => 14 x 14 x 480
        )

        self.layer4 = nn.Sequential(
            InceptionV1(480, 192, 96, 208, 16, 48, 64),  # 14 x 14 x 480 => 14 x 14 x 512
            InceptionV1(512, 160, 112, 224, 24, 64, 64),  # 14 x 14 x 512 => 14 x 14 x 512
            InceptionV1(512, 128, 128, 256, 24, 64, 64),  # 14 x 14 x 512 => 14 x 14 x 512
            InceptionV1(512, 112, 144, 288, 32, 64, 64),  # 14 x 14 x 512 => 14 x 14 x 528
            InceptionV1(528, 256, 160, 320, 32, 128, 128),  # 14 x 14 x 528, 14 x 14 x 832
            nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 14 x 14 x 832 => 7 x 7 x 832
        )

        self.layer5 = nn.Sequential(
            InceptionV1(832, 256, 160, 320, 32, 128, 128),  # 7 x 7 x 832 => 7 x 7 x 832
            InceptionV1(832, 384, 192, 384, 48, 128, 128),  # 7 x 7 x 832 => 7 x 7 x 1024
            nn.AvgPool2d(7, stride=1, ceil_mode=True)  # 7 x 7 x 1024 => 1 x 1 x 1024
        )

        self.layer6 = nn.Sequential(
            nn.Dropout2d(0.4),
            nn.Linear(1024, 1000),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 1024)
        x = self.layer6(x)
        return x


if __name__ == '__main__':
    googlenet = GoogleNet()
    train = True
    epoch = 10
    if train:
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(googlenet.parameters(), lr=0.001)

        for i in range(epoch):
            running_loss = 0.0

            for j, data in enumerate(trainloader, 0):
                images, labels = data
                opt.zero_grad()
                outputs = googlenet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss
                if j % 5 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, j + 1, running_loss / 5))
                    running_loss = 0.0

        print('Trainning Completed')
        torch.save(googlenet.state_dict(), PATH)
    else:
        googlenet.load_state_dict(torch.load(PATH))
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = googlenet(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
