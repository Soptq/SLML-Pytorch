import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.random.rand(256)
    noise = np.random.rand(256) / 4
    y = x * 5 + 7 + noise  # y = 5 * x + 7
    # plt.scatter(x, y)
    # plt.show()

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimation = optim.SGD(model.parameters(), lr=0.01)

    epochs = 3000

    x_train = x.reshape(-1, 1).astype('float32')
    y_train = y.reshape(-1, 1).astype('float32')

    for i in range(epochs):
        inputs = torch.from_numpy(x_train)
        labels = torch.from_numpy(y_train)

        outputs = model(inputs)

        optimation.zero_grad()

        loss = criterion(outputs, labels)

        loss.backward()
        optimation.step()
        if i % 100 == 0:
            print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))

    w, b = model.parameters()
    print(w.item(), b.item())
