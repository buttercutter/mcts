# This file is the test/inference coding
# See https://github.com/bsamseth/tictacNET/issues/2
# for a description of the inputs and output of the neural network

import numpy as np
import torch
import random
from Net import Net


CROSS, NOUGHT = 0, 1
PLAYERS = [CROSS, NOUGHT]

# Winning patterns encoded in bit patterns.
# E.g. three in a row in the top row is
#   448 = 0b111000000
WINNING_PATTERNS = [448, 56, 7, 292, 146, 73, 273, 84]  # Row  # Columns  # Diagonals


# initial game config
TOTAL_NUM__OF_BOXES = 9  # 3x3 tic-tac-toe
random_player_start_turn = random.randint(CROSS, NOUGHT)
cross_positions = '000000000'
nought_positions = '000000000'
player_turn = random_player_start_turn
model_input = cross_positions + nought_positions + str(player_turn)
next_move = 99999999999999  # just for initialization
next_move_probabilities = np.zeros(TOTAL_NUM__OF_BOXES)  # 9 boxes choice
predicted_score = np.zeros(3)  # loss, draw, win
out = [next_move_probabilities, predicted_score]


def binary_to_string(input_binary):
    mask = 0b1
    output = ""
    for i in range(TOTAL_NUM__OF_BOXES):
        if mask & input_binary:
            output += '1'
        else:
            output += '0'
        mask <<= 1
    return output[::-1]


def update_move(_next_move, _next_move_probabilities, _player_turn, _cross_positions, _nought_positions):
    print("we will update the next_move accordingly inside this function")

    _next_move = np.binary_repr(_next_move_probabilities.argmax())

    # What if the square box (next_move) is already filled by the other player ?
    # then we will have to go for the next less preferred next_move
    # During actual project in later stage, we will use Monte-Carlo Tree Search
    # instead of the following logic
    cross_positions_str = binary_to_string(int(_cross_positions, 2))
    nought_positions_str = binary_to_string(int(_nought_positions, 2))
    _next_move_in_integer = int(_next_move, 2)

    print("next_move in integer = ", _next_move_in_integer)
    print("cross_positions_str = ", cross_positions_str)
    print("nought_positions_str = ", nought_positions_str)

    # if opponent player had filled the square box position
    #    OR the same player had filled the same square box position
    if ((_player_turn == CROSS) and (nought_positions_str[_next_move_in_integer] == '1')) or \
       ((_player_turn == NOUGHT) and (cross_positions_str[_next_move_in_integer] == '1')) or \
       ((_player_turn == CROSS) and (cross_positions_str[_next_move_in_integer] == '1')) or \
       ((_player_turn == NOUGHT) and (nought_positions_str[_next_move_in_integer] == '1')):

        print("going for second preferred next move")
        print("next_move_probabilities = ", _next_move_probabilities)
        _next_move_probabilities[0, _next_move_in_integer] = 0  # makes way for less preferred next_move
        print("after setting certain bit to 0, next_move_probabilities = ", _next_move_probabilities)
        _next_move = np.binary_repr(_next_move_probabilities.argmax())

        # checks again whether this less preferred next_move had already been played before
        _next_move = update_move(_next_move, _next_move_probabilities, _player_turn,
                                 _cross_positions, _nought_positions)

    return _next_move


def players_have_winning_patterns(_cross_positions, _nought_positions):
    # needs to match every bits in the WINNING_PATTERNS
    if any(
            np.bitwise_and(win, int(_cross_positions, 2)) == win or
            np.bitwise_and(win, int(_nought_positions, 2)) == win
            for win in WINNING_PATTERNS
    ):
        return 1

    else:
        return 0


PATH = './tictactoe_net.pth'

# Load
model = torch.load(PATH)
model.eval()

USE_CUDA = torch.cuda.is_available()

# while (cross_positions != WINNING_PATTERNS) | (nought_positions != WINNING_PATTERNS):
while players_have_winning_patterns(cross_positions, nought_positions) == 0:  # game is still ON
    if USE_CUDA:
        out_policy, out_value = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ).cuda())

    else:
        out_policy, out_value  = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ))

    print("out_policy = ", out_policy)
    print("out_value = ", out_value)
    next_move_probabilities = out_policy

    # updates next_move
    next_move = update_move(next_move, next_move_probabilities, player_turn, cross_positions, nought_positions)

    next_move_in_integer = int(next_move, 2)
    print("Confirmed next_move = ", next_move_in_integer)

    # updates cross_positions or NAUGHT_POSITIONS (based on next_move output from NN)
    # depending on which player turn
    if player_turn == CROSS:
        # bitwise OR (cross_positions, next_move)
        cross_positions = binary_to_string(int(cross_positions, 2) |
                                           (1 << (TOTAL_NUM__OF_BOXES-next_move_in_integer-1)))

    else:
        # bitwise OR (nought_positions, next_move)
        nought_positions = binary_to_string(int(nought_positions, 2) |
                                            (1 << (TOTAL_NUM__OF_BOXES-next_move_in_integer-1)))

    print("cross_positions = ", cross_positions)
    print("nought_positions = ", nought_positions)

    # flips turn for next player, use a NOT operator since there are only 2 players
    if player_turn == CROSS:
        player_turn = NOUGHT

    else:
        player_turn = CROSS

    print("player_turn = ", player_turn)

    # updates model_input for next player turn
    model_input = cross_positions + nought_positions + str(player_turn)

    print("model_input = ", model_input)
    print("\n")
