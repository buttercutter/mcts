# This file is the test/inference coding
# See https://github.com/bsamseth/tictacNET/issues/2
# for a description of the inputs and output of the neural network

import numpy as np
import torch
import random
from Net import Net


def binary_to_string(input_binary):
    mask = 0b1
    output = ""
    for i in range(9):
        if mask & input_binary:
            output += '1'
        else:
            output += '0'
        mask <<= 1
    return output[::-1]


CROSS, NOUGHT = 0, 1
PLAYERS = [CROSS, NOUGHT]

# Winning patterns encoded in bit patterns.
# E.g. three in a row in the top row is
#   448 = 0b111000000
WINNING_PATTERNS = [448, 56, 7, 292, 146, 73, 273, 84]  # Row  # Columns  # Diagonals

PATH = './tictactoe_net.pth'

# Load
model = torch.load(PATH)
model.eval()

USE_CUDA = torch.cuda.is_available()

# initial game config
random_player_start_turn = random.randint(CROSS, NOUGHT)
CROSS_POSITIONS = '000000000'
NOUGHT_POSITIONS = '000000000'
player_turn = random_player_start_turn
model_input = CROSS_POSITIONS + NOUGHT_POSITIONS + str(player_turn)
next_move_probabilities = np.zeros(9)  # 9 boxes choice
predicted_score = np.zeros(3)  # loss, draw, win
out = [next_move_probabilities, predicted_score]

while (CROSS_POSITIONS != WINNING_PATTERNS) | (NOUGHT_POSITIONS != WINNING_PATTERNS):  # game is still ON
    if USE_CUDA:
        out = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ).cuda())

    else:
        out = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ))

    print(out)
    next_move_probabilities = out[0]

    # updates next_move
    next_move = np.binary_repr(next_move_probabilities.argmax())
    print("next_move = ", next_move)

    # updates CROSS_POSITIONS or NAUGHT_POSITIONS (based on next_move output from NN)
    # depending on which player turn
    if player_turn == CROSS:
        # bitwise OR (CROSS_POSITIONS, next_move)
        CROSS_POSITIONS = binary_to_string(int(CROSS_POSITIONS, 2) | int(next_move, 2))

    else:
        # bitwise OR (NOUGHT_POSITIONS, next_move)
        NOUGHT_POSITIONS = binary_to_string(int(NOUGHT_POSITIONS, 2) | int(next_move, 2))

    print("CROSS_POSITIONS = ", CROSS_POSITIONS)
    print("NAUGHT_POSITIONS = ", NOUGHT_POSITIONS)

    # flips turn for next player, use a NOT operator since there are only 2 players
    if player_turn == CROSS:
        player_turn = NOUGHT

    else:
        player_turn = CROSS

    print("player_turn = ", player_turn)

    # updates model_input for next player turn
    model_input = CROSS_POSITIONS + NOUGHT_POSITIONS + str(player_turn)

    print("model_input = ", model_input)
