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


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # 3 x 224 x 224 => 64 x 224 x 224
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 64 x 224 x 224 => 64 x 224 x 224
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 64 x 112 x 112 => 128 x 112 x 112
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  # 128 x 112 x 112 => 128 x 112 x 112
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 128 x 56 x 56 => 256 x 56 x 56
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)  # 256 x 56 x 56 => 256 x 56 x 56
        self.conv7 = nn.Conv2d(256, 256, 3, stride=1, padding=1)  # 256 x 56 x 56 => 256 x 56 x 56
        self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)  # 256 x 28 x 28 => 512 x 28 x 28
        self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)  # 512 x 28 x 28 => 512 x 28 x 28

        self.mp = nn.MaxPool2d(2, stride=2)

        self.do = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.mp(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.mp(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.mp(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.mp(F.relu(self.conv9(F.relu(self.conv9(F.relu(self.conv8(x)))))))
        x = self.mp(F.relu(self.conv9(F.relu(self.conv9(F.relu(self.conv9(x)))))))
        x = x.view(-1, 512 * 7 * 7)
        x = self.do(F.relu(self.fc1(x)))
        x = self.do(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    vggnet = VGG()
    train = True
    epoch = 10
    if train:
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(vggnet.parameters(), lr=0.001)

        for i in range(epoch):
            running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                images, labels = data
                opt.zero_grad()
                outputs = vggnet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss
                if j % 2 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, j + 1, running_loss / 2))
                    running_loss = 0.0

        print('Trainning Completed')
        torch.save(vggnet.state_dict(), PATH)
    else:
        vggnet.load_state_dict(torch.load(PATH))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testset:
                images, labels = data
                outputs = vggnet(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
