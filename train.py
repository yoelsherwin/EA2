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
        self.fc3 = nn.Linear(80, 80)
        self.bn3 = nn.BatchNorm1d(80)
        self.fc4 = nn.Linear(80, 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.fc5 = nn.Linear(100, 80)
        self.bn5 = nn.BatchNorm1d(80)
        self.fc6 = nn.Linear(80, 30)
        self.bn6 = nn.BatchNorm1d(30)
        self.fc7 = nn.Linear(30, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))


        x = self.fc7(x)
        # x = self.softmax(x)

        return x

def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 25

    for epoch in range(epochs):
        sum_loss = 0
        num_items = 0
        num_corrects = 0

        #optimizer.zero_grad()
        # for x, y in data:
        for x, y in data:
            optimizer.zero_grad()
            y_hat = model(x)

            loss = loss_fn(y_hat, y)
            # got = []
            # target = []
            # got.append(Fbeta)
            # target.append(1)
            # loss = loss_fn(torch.tensor(got), torch.tensor(target))
            #loss = 1 - Fbeta
            loss.backward()

            #sum_loss += loss.item()
            #num_items += len(y)

            #argmax = y_hat.argmax(dim=1)
            #corrects = y == argmax
            #corrects = corrects.sum()
            # print(corrects)
            # print(corrects.size())
            # print(corrects.sum())
            #num_corrects += corrects.sum()

            optimizer.step()
        #optimizer.step()