import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from collections import Counter

# part of the code is referenced from
# https://github.com/bsamseth/tictacNET/blob/master/tictacnet.py and
# https://github.com/bsamseth/tictacNET/blob/master/tictactoe-data.csv

df = pd.read_csv("tictactoe-data.csv")
print("Scores:", Counter(df["score"]))

num_of_board_features = 18
player_turn = -2  # second to last column of the csv file

# Input is all the board features (2x9 squares) plus the turn.
board_features_and_turn = df.iloc[:, list(range(num_of_board_features)) + [player_turn]]

num_of_possible_moves = 9

# Target variables are the possible move squares.
# moves = df.iloc[:, list(range(num_of_board_features, num_of_board_features+num_of_possible_moves))]


# To predict score instead, use this as the target:
score = pd.get_dummies(df['score'])


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(32, 3)
        )

    def forward(self, x):

        out = self.linears(x)
        return out


net = Net()
print(net)

params = list(net.parameters())
print(len(params))

# for x in range(10):
#    print(params[x].size())  # conv1's .weight

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lossFunction = nn.MSELoss()

num_epochs = 100
for epoch in range(num_epochs):
    loss_ = 0

    # See https://github.com/bsamseth/tictacNET/issues/2
    # for a description of the inputs and output of the neural network
    train_loader = list(zip(
        torch.from_numpy(np.array(board_features_and_turn, dtype='float32')[:, np.newaxis]),
        torch.from_numpy(np.array(score, dtype='float32')[:, np.newaxis]),
    ))

    for board_features_and_turn, score in train_loader:
        # Forward Pass
        output = net(board_features_and_turn)
        # Loss at each iteration by comparing to target(score)
        loss = lossFunction(output, score)

        # Backpropogating gradient of loss
        optimizer.zero_grad()
        loss.backward()

        # Updating parameters(weights and bias)
        optimizer.step()

        loss_ += loss.item()
    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(score)))

print('Finished Training')

PATH = './tictactoe_net.pth'
torch.save(net.state_dict(), PATH)
