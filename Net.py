import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


TEST_DATASET_RATIO = 0.05  # 5 percent of the dataset is dedicated for testing purpose
NUM_OF_BOARD_FEATURES = 18
PLAYER_TURN_COLUMN = -2  # second to last column of the csv file
SCORE_COLUMN = -1  # last column of the csv file
NUM_OF_BOARD_FEATURES_AND_TURN = NUM_OF_BOARD_FEATURES + 1  # add 1 because of player turn
NUM_OF_POSSIBLE_MOVES = 9  # a normal 3x3 tic-tac-toe has 9 input boxes
NUM_OF_POSSIBLE_SCORES = 3  # -1, 0, 1 == loss, draw, win
POSSIBLE_SCORES = [-1, 0, 1]
SIZE_OF_HIDDEN_LAYERS = 512
NUM_EPOCHS = 6000
LEARNING_RATE = 0.7
MOMENTUM = 0.9


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(NUM_OF_BOARD_FEATURES_AND_TURN, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, NUM_OF_POSSIBLE_MOVES)
        )

        self.linears2 = nn.Sequential(
            nn.Linear(NUM_OF_BOARD_FEATURES_AND_TURN, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, SIZE_OF_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(SIZE_OF_HIDDEN_LAYERS, NUM_OF_POSSIBLE_SCORES)
        )

    def forward(self, x):
        policy = self.linears(x)
        value = self.linears2(x)

        return policy, value  # move, score


USE_CUDA = torch.cuda.is_available()


def train():
    # part of the code is referenced from
    # https://github.com/bsamseth/tictacNET/blob/master/tictacnet.py and
    # https://github.com/bsamseth/tictacNET/blob/master/tictactoe-data.csv

    df = pd.read_csv("tictactoe-data.csv")
    print("Scores:", Counter(df["score"]))

    # Input is all the board features (2x9 squares) plus the turn.
    board_features_and_turn = df.iloc[:, list(range(NUM_OF_BOARD_FEATURES)) + [PLAYER_TURN_COLUMN]]

    # To predict score instead, use this as the target:
    # score = pd.get_dummies(df['score'])
    # print(score)

    # split into training dataset (80%) and validation dataset (20%)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html and
    # https://scikit-learn.org/stable/glossary.html#term-random_state show that splitting is randomized.
    # Since the dataset involving 1 input and 2 outputs, need to combine the 2 outputs first before splitting
    # in order to preserve data alignment

    # Target variables are the possible move squares as well as the predicted output scores
    moves_score = df.iloc[:, list(range(NUM_OF_BOARD_FEATURES, NUM_OF_BOARD_FEATURES + NUM_OF_POSSIBLE_MOVES)) +
                          [SCORE_COLUMN]]
    # print(moves_score)

    board_train, board_test, moves_score_train, moves_score_test = \
        train_test_split(board_features_and_turn, moves_score, test_size=TEST_DATASET_RATIO)
    # print(board_test)

    moves_test = moves_score_test.iloc[:, list(range(0, NUM_OF_POSSIBLE_MOVES))]
    score_test = moves_score_test.iloc[:, [NUM_OF_POSSIBLE_MOVES]]

    moves_train = moves_score_train.iloc[:, list(range(0, NUM_OF_POSSIBLE_MOVES))]
    score_train = moves_score_train.iloc[:, [NUM_OF_POSSIBLE_MOVES]]

    print(len(score_train))
    print(len(score_test))

    net = Net()
    if USE_CUDA:
        net = net.cuda()
    print(net)

    params = list(net.parameters())
    print(len(params))

    # for x in range(10):
    #    print(params[x].size())  # conv1's .weight

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()

    TRAIN_BATCH_SIZE = int(len(moves_score)*(1-TEST_DATASET_RATIO))

    for epoch in range(NUM_EPOCHS):
        loss_ = 0

        # See https://github.com/bsamseth/tictacNET/issues/2
        # for a description of the inputs and output of the neural network
        train_loader = zip(np.array(board_train, dtype='float32'),
                           np.array(moves_train, dtype='float32'),
                           np.array(score_train, dtype='float32'))
        train_loader = torch.utils.data.DataLoader(
            list(train_loader),
            batch_size=TRAIN_BATCH_SIZE,
        )

        for _board_features_and_turn, move, _score in train_loader:
            if USE_CUDA:
                _board_features_and_turn = _board_features_and_turn.cuda()
                move = move.cuda()
                _score = _score.cuda()

            # Forward Pass
            policy_output, value_output = net(_board_features_and_turn)

            # Since both policy_output and value_output are of continuous probability nature,
            # we need to change them to discrete number for loss_function() computation
            policy_output_discrete = torch.zeros(len(_score), NUM_OF_POSSIBLE_MOVES, requires_grad=True)
            if USE_CUDA:
                policy_output_discrete = policy_output_discrete.cuda()

            for topk_index in range(len(_score)):  # functionally equivalent to softmax()
                policy_output_discrete[topk_index][policy_output.topk(1).indices[topk_index]] = 1

            # substract 1 because score is one of these [-1, 0, 1] values
            value_output_discrete = torch.topk(value_output, 1).indices - 1

            # Loss at each iteration by comparing to target(moves)
            loss1 = loss_function(policy_output_discrete, move)
            # Loss at each iteration by comparing to target(score)
            loss2 = loss_function(value_output_discrete, _score)

            loss = loss1 + loss2

            # Backpropagating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer.step()

            loss_ += loss.item()
        print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(moves_train)))

    print('Finished Training')

    path = './tictactoe_net.pth'
    torch.save(net, path)

    print("############################################")
    print("Doing train_accuracy check")

    train_correct = 0
    train_total = 0

    train_loader = zip(np.array(board_train, dtype='float32'),
                       np.array(moves_train, dtype='float32'),
                       np.array(score_train, dtype='float32'))
    train_loader = torch.utils.data.DataLoader(
        list(train_loader),
        batch_size=TRAIN_BATCH_SIZE,
    )

    with torch.no_grad():
        for _board_train, _moves_train, _score_train in train_loader:
            if USE_CUDA:
                _board_train = _board_train.cuda()
                _moves_train = _moves_train.cuda()
                _score_train = _score_train.cuda()

            model_input = _board_train
            _policy_output, _value_output = net(model_input)
            predicted = torch.argmax(_policy_output, 1)

            print("_policy_output = ", _policy_output)
            print("predicted = ", predicted)
            print("_moves_train = ", _moves_train)

            for train_index in range(len(_moves_train)):
                print("move testing for train_index = ", train_index)

                print("_moves_train[train_index][predicted[train_index]] = ",
                      _moves_train[train_index][predicted[train_index]], '\n')

                if _moves_train[train_index][predicted[train_index]]:
                    print("predicted == _moves_train")
                    train_correct = train_correct + 1

            train_total = train_total + len(_moves_train)

    print('Accuracy of the network on train move: %d %%' % (
        100 * train_correct / train_total))

    print("############################################")
    print("Doing test_accuracy check")

    # validate the trained NN model for both predicted recommended move
    # and its corresponding predicted score
    TEST_BATCH_SIZE = int(len(moves_score)*TEST_DATASET_RATIO)  # 4520*0.8

    test_correct = 0
    test_total = 0

    test_loader = zip(np.array(board_test, dtype='float32'),
                      np.array(moves_test, dtype='float32'),
                      np.array(score_test, dtype='float32'))
    test_loader = torch.utils.data.DataLoader(
        list(test_loader),
        batch_size=TEST_BATCH_SIZE,
    )
    print(test_loader)

    with torch.no_grad():
        for _board_test, _moves_test, _score_test in test_loader:
            if USE_CUDA:
                _board_test = _board_test.cuda()
                _moves_test = _moves_test.cuda()
                _score_test = _score_test.cuda()

            model_input = _board_test
            _policy_output, _value_output = net(model_input)
            predicted = torch.argmax(_policy_output, 1)

            # print("_policy_output = ", _policy_output)
            # print("predicted = ", predicted)
            # print("_moves_test = ", _moves_test)

            for test_index in range(len(_moves_test)):
                # print("move testing for test_index = ", test_index)

                # print("_moves_test[test_index][predicted[test_index]] = ",
                #      _moves_test[test_index][predicted[test_index]], '\n')

                if _moves_test[test_index][predicted[test_index]]:
                    # print("predicted == _moves_test")
                    test_correct = test_correct + 1

            test_total = test_total + len(_moves_test)

    print('Accuracy of the network on test move: %d %%' % (
        100 * test_correct / test_total))

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for _board_test, _moves_test, _score_test in test_loader:
            if USE_CUDA:
                _board_test = _board_test.cuda()
                _moves_test = _moves_test.cuda()
                _score_test = _score_test.cuda()

            model_input = _board_test
            _policy_output, _value_output = net(model_input)
            predicted = torch.argmax(_value_output, 1)

            # print("_value_output = ", _value_output)
            # print("predicted = ", predicted)
            # print("_score_test = ", _score_test)

            for test_index in range(len(_score_test)):
                # print("move testing for test_index = ", test_index)

                # print("_score_test[test_index][predicted[test_index]] = ",
                #      _score_test[test_index][predicted[test_index]], '\n')

                if _score_test[test_index] == POSSIBLE_SCORES[predicted[test_index]-1]:
                    # print("predicted == _score_test")
                    test_correct = test_correct + 1

            test_total = test_total + len(_score_test)

    print('Accuracy of the network on test score: %d %%' % (
        100 * test_correct / test_total))


if __name__ == "__main__":
    train()
