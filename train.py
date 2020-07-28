import torch
import torch.nn as nn
import torch.nn.functional as F
from evolution import THRESHOLD as THRESHOLD

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

        return x

def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1

    for epoch in range(epochs):
        #TP = TN = FP = FN = 0
        for x, y in data:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            # res = model(x)
            # ans = []
            # for i in range(len(res)):
            #     if res[i][0] + THRESHOLD < res[i][1]:
            #         ans.append(1)
            #     else:
            #         ans.append(0)
            # res = ans
            # for i in range(len(res)):
            #     if res[i] == y[i] and y[i] == 1:
            #         TP += 1
            #     elif res[i] == y[i] and y[i] == 0:
            #         TN += 1
            #     elif res[i] == 0 and y[i] == 1:
            #         FN += 1
            #     else:
            #         FP += 1
            # if ((TP + FP) == 0):
            #     precision = 0
            # else:
            #     precision = TP / (TP + FP)
            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            # if ((TP + FN) == 0):
            #     recall = 0
            # else:
            #     recall = TP / (TP + FN)
            # # if ((0.0156 * precision + recall) != 0):
            # if ((0.0625 * precision + recall) != 0):
            #     Fbeta = (1.0625 * precision * recall) / (0.0625 * precision + recall)
            # #   Fbeta = (1.0156 * precision * recall) / (0.0156 * precision + recall)
            # else:
            #     Fbeta = 0
            #
            # got = []
            # target = []
            # got.append(Fbeta)
            # target.append(1)
            # loss = loss_fn(torch.tensor(got), torch.tensor(target))

            loss.backward()
            optimizer.step()