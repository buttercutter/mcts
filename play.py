# This file is the test/inference coding
# See https://github.com/bsamseth/tictacNET/issues/2
# for a description of the inputs and output of the neural network

import numpy as np
import torch
import random
from Net import Net


CROSS, NOUGHT = 0, 1
PLAYERS = [CROSS, NOUGHT]
TOTAL_NUM__OF_BOXES = 9  # 3x3 tic-tac-toe


# Winning patterns encoded in bit patterns.
# E.g. three in a row in the top row is
#   448 = 0b111000000
WINNING_PATTERNS = [448, 56, 7, 292, 146, 73, 273, 84]  # Row  # Columns  # Diagonals
TIE_DRAW = 0
CROSS_IS_WINNER = 1
NOUGHT_IS_WINNER = 2
NO_WINNING_PLAYER_YET = 3


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


def player_cross_has_winning_patterns(_cross_positions):
    # needs to match every bits in the WINNING_PATTERNS
    if any(
            np.bitwise_and(win, int(_cross_positions, 2)) == win
            for win in WINNING_PATTERNS
    ):
        return 1

    else:
        return 0


def player_nought_has_winning_patterns(_nought_positions):
    # needs to match every bits in the WINNING_PATTERNS
    if any(
            np.bitwise_and(win, int(_nought_positions, 2)) == win
            for win in WINNING_PATTERNS
    ):
        return 1

    else:
        return 0


def initialize():
    # initial game config
    random_player_start_turn_ = random.randint(CROSS, NOUGHT)
    cross_positions_ = '000000000'
    nought_positions_ = '000000000'
    player_turn_ = random_player_start_turn_
    model_input_ = cross_positions_ + nought_positions_ + str(player_turn_)
    # next_move_ = 99999999999999  # just for initialization
    # next_move_probabilities = np.zeros(TOTAL_NUM__OF_BOXES)  # 9 boxes choice
    # predicted_score = np.zeros(3)  # loss, draw, win
    # out = [next_move_probabilities, predicted_score]

    trained_model_path = './tictactoe_net.pth'

    # Load
    model_ = torch.load(trained_model_path)
    model_.eval()

    return cross_positions_, nought_positions_, model_, model_input_, player_turn_


def play(using_mcts, best_child_node, model, model_input, player_turn, cross_positions, nought_positions):
    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        out_policy, out_value = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ).cuda())

    else:
        out_policy, out_value = model(torch.from_numpy(
            np.array([int(v) for v in model_input], dtype='float32')[np.newaxis]
        ))

    print("out_policy = ", out_policy)
    print("out_value = ", out_value)
    next_move_probabilities = out_policy

    if using_mcts:  # will determine next_move according to highest PUCT values of child nodes
        next_move = best_child_node

    else:
        # updates next_move
        next_move = torch.argmax(next_move_probabilities)

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

    print("player_turn = ", player_turn)

    # updates model_input for next player turn
    model_input = cross_positions + nought_positions + str(player_turn)

    print("model_input = ", model_input)
    print("\n")

    return out_value, cross_positions, nought_positions, model_input


if __name__ == '__main__':

    print("standalone inference coding")
    game_is_on = 1
    num_of_play_rounds = 0
    _cross_positions, _nought_positions, _model, _model_input, _player_turn = initialize()

    # while (cross_positions != WINNING_PATTERNS) | (nought_positions != WINNING_PATTERNS):
    while game_is_on:  # game is still ON
        num_of_play_rounds = num_of_play_rounds + 1

        __out_value, __cross_positions, __nought_positions, _model_input \
            = play(0, 0, _model, _model_input, _player_turn, _cross_positions, _nought_positions)

        out_score = torch.argmax(__out_value)
        game_is_on = num_of_play_rounds < TOTAL_NUM__OF_BOXES

        cross_had_won = player_cross_has_winning_patterns(__cross_positions)
        nought_had_won = player_nought_has_winning_patterns(__nought_positions)

        if (game_is_on == 0) & (cross_had_won == 0) & (nought_had_won == 0):
            print("game finished with draw")

        if cross_had_won:
            print("game finished with player CROSS being the winner")

        if nought_had_won:
            print("game finished with player NOUGHT being the winner")

        _cross_positions = __cross_positions
        _nought_positions = __nought_positions

        # switches player turn after each step
        if _player_turn == CROSS:
            _player_turn = NOUGHT

        else:
            _player_turn = CROSS

    game_is_on = 0
    print("game finished")

else:  # executed from mcts.py

    print("using mcts")
    _cross_positions = 0
    _nought_positions = 0
    _model = 0
    _model_input = 0
    _player_turn = 0
    num_of_play_rounds = 0
    game_is_on = 0  # initialized to 0 because game has not started yet
    out_score = 0

    def mcts_play(is_mcts_in_simulate_stage=0, ongoing_game=0, best_child_node=0):

        global _cross_positions
        global _nought_positions
        global _model
        global _model_input
        global _player_turn
        global num_of_play_rounds
        global game_is_on
        global out_score

        if is_mcts_in_simulate_stage:
            print("is_mcts_in_simulate_stage")

            if ongoing_game == 0:  # first step of the game
                print("ongoing_game == 0")
                game_is_on = 1
                num_of_play_rounds = 0
                _cross_positions, _nought_positions, _model, _model_input, _player_turn = initialize()

            print("ongoing_game == 1")
            num_of_play_rounds = num_of_play_rounds + 1
            print("num_of_play_rounds = ", num_of_play_rounds)

            __out_value, __cross_positions, __nought_positions, _model_input \
                = play(1, best_child_node, _model, _model_input, _player_turn, _cross_positions, _nought_positions)

            out_score = torch.argmax(__out_value)
            game_is_on = num_of_play_rounds < TOTAL_NUM__OF_BOXES

            cross_had_won = player_cross_has_winning_patterns(__cross_positions)
            nought_had_won = player_nought_has_winning_patterns(__nought_positions)

            if (game_is_on == 0) & (cross_had_won == 0) & (nought_had_won == 0):
                print("game finished with draw")
                return TIE_DRAW

            if cross_had_won:
                return CROSS_IS_WINNER

            if nought_had_won:
                return NOUGHT_IS_WINNER

            _cross_positions = __cross_positions
            _nought_positions = __nought_positions

            # switches player turn after each step
            print("switches player turn")
            if _player_turn == CROSS:
                _player_turn = NOUGHT

            else:
                _player_turn = CROSS

        return NO_WINNING_PLAYER_YET
