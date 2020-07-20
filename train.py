import torch
import torch.nn as nn
import torch.nn.functional as F
import create_params
import numpy as np
import data_loader as dl


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(16, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.fc3 = nn.Linear(80, 50)
        self.bn3 = nn.BatchNorm1d(50)

        self.fc4 = nn.Linear(50, 40)
        self.bn4 = nn.BatchNorm1d(40)
        self.fc5 = nn.Linear(40, 20)
        self.bn5 = nn.BatchNorm1d(20)
        self.fc6 = nn.Linear(20, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))


        x = self.fc6(x)
        # x = self.softmax(x)

        return x

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1000

    for epoch in range(epochs):
        sum_loss = 0
        num_items = 0
        num_corrects = 0

        optimizer.zero_grad()
        for x, y in dl.train_loader:
            y_hat = model(x)

            loss = loss_fn(y_hat, y)
            loss.backward()

            sum_loss += loss.item()
            num_items += len(y)

            argmax = y_hat.argmax(dim=1)
            corrects = y == argmax
            corrects = corrects.sum()
            # print(corrects)
            # print(corrects.size())
            # print(corrects.sum())
            num_corrects += corrects.sum()

        optimizer.step()

        print(f"average loss: {sum_loss / num_items}")
        print(f"accuracy: {float(num_corrects) / float(num_items)}")
