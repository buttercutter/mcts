import torch
import torch.utils.data
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

num_of_possible_moves = 9  # a normal 3x3 tic-tac-toe has 9 input boxes
num_of_possible_scores = 3  # -1, 0, 1 == loss, draw, win

# Target variables are the possible move squares.
moves = df.iloc[:, list(range(num_of_board_features, num_of_board_features + num_of_possible_moves))]

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

        return policy, value  # move, score


USE_CUDA = torch.cuda.is_available()


def train():
    net = Net()
    if USE_CUDA:
        net = net.cuda()
    print(net)

    params = list(net.parameters())
    print(len(params))

    # for x in range(10):
    #    print(params[x].size())  # conv1's .weight

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()

    num_epochs = 200
    for epoch in range(num_epochs):
        loss_ = 0

        # See https://github.com/bsamseth/tictacNET/issues/2
        # for a description of the inputs and output of the neural network
        train_loader = zip(np.array(board_features_and_turn, dtype='float32'),
                           np.array(moves, dtype='float32'),
                           np.array(score, dtype='float32'))
        train_loader = torch.utils.data.DataLoader(
            list(train_loader),
            batch_size=32,
        )

        for _board_features_and_turn, move, _score in train_loader:
            if USE_CUDA:
                _board_features_and_turn = _board_features_and_turn.cuda()
                move = move.cuda()
                _score = _score.cuda()

            # Forward Pass
            policy_output, value_output = net(_board_features_and_turn)
            # Loss at each iteration by comparing to target(moves)
            loss1 = loss_function(policy_output, move)
            # Loss at each iteration by comparing to target(score)
            loss2 = loss_function(value_output, _score)

            loss = loss1 + loss2

            # Backpropogating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer.step()

            loss_ += loss.item()
        print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(score)))

    print('Finished Training')

    path = './tictactoe_net.pth'
    torch.save(net, path)


if __name__ == "__main__":
    train()
