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
num_of_board_features_and_turn = num_of_board_features + 1  # add 1 because of player turn

# Input is all the board features (2x9 squares) plus the turn.
board_features_and_turn = df.iloc[:, list(range(num_of_board_features)) + [player_turn]]

num_of_possible_moves = 9   # a normal 3x3 tic-tac-toe has 9 input boxes
num_of_possible_scores = 3  # -1, 0, 1 == loss, draw, win

# Target variables are the possible move squares.
moves = df.iloc[:, list(range(num_of_board_features, num_of_board_features+num_of_possible_moves))]


# To predict score instead, use this as the target:
score = pd.get_dummies(df['score'])


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(num_of_board_features_and_turn, 128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(32, num_of_possible_moves)
        )

        self.linears2 = nn.Sequential(
            nn.Linear(num_of_board_features_and_turn, 128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(32, num_of_possible_scores)
        )

    def forward(self, x):

        policy = self.linears(x)
        value = self.linears2(x)

        return policy, value  # score, move


net = Net()
print(net)

params = list(net.parameters())
print(len(params))

# for x in range(10):
#    print(params[x].size())  # conv1's .weight

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lossFunction = nn.MSELoss()

num_epochs = 500
for epoch in range(num_epochs):
    loss_ = 0

    # See https://github.com/bsamseth/tictacNET/issues/2
    # for a description of the inputs and output of the neural network
    train_loader = list(zip(
        torch.from_numpy(np.array(board_features_and_turn, dtype='float32')[:, np.newaxis]),
        torch.from_numpy(np.array(moves, dtype='float32')[:, np.newaxis]),
        torch.from_numpy(np.array(score, dtype='float32')[:, np.newaxis]),
    ))

    for board_features_and_turn, moves, score in train_loader:
        # Forward Pass
        policy_output, value_output = net(board_features_and_turn)
        # Loss at each iteration by comparing to target(moves)
        loss1 = lossFunction(policy_output, moves)
        # Loss at each iteration by comparing to target(score)
        loss2 = lossFunction(value_output, score)

        loss = loss1 + loss2

        # Backpropogating gradient of loss
        optimizer.zero_grad()
        loss.backward()

        # Updating parameters(weights and bias)
        optimizer.step()

        loss_ += loss.item()
    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(score)))

print('Finished Training')

PATH = './tictactoe_net.pth'
torch.save(net, PATH)
