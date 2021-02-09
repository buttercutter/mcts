import random
import numpy as np
from Net import Net
import play

import sys
sys.setrecursionlimit(100000)  # to temporarily solve Recursion Depth Limit issue

# Reference :
# https://www.reddit.com/r/learnmachinelearning/comments/fmx3kv/empirical_example_of_mcts_calculation_puct_formula/

# PUCT formula : https://colab.research.google.com/drive/14v45o1xbfrBz0sG3mHbqFtYz_IrQHLTg#scrollTo=1VeRCpCSaHe3

# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
cfg_puct = np.sqrt(2)  # to balance between exploitation and exploration
puct_array = []  # stores puct ratio for every child nodes for argmax()


# determined by PUCT formula
def find_best_path(parent):
    print("find_best_path()")
    if len(parent.nodes) == 0:
        return 0

    for N in parent.nodes:
        puct_array.append(N.puct)

    max_index = np.argmax(puct_array)
    puct_array.clear()  # resets the list so that other paths could reuse it

    #  leaf node has 0 child node
    is_leaf_node = (len(parent.nodes[max_index].nodes) == 0)
    if is_leaf_node:
        return max_index

    else:
        return -1


# for play.py inference coding
is_simulation_stage = None  # initialized to None because game had not started yet


def is_mcts_in_simulate_stage():
    return is_simulation_stage


class Mcts:
    def __init__(self, parent):
        # https://www.tutorialspoint.com/python_data_structure/python_tree_traversal_algorithms.htm
        # https://www.geeksforgeeks.org/sum-parent-nodes-child-node-x/

        self.parent = parent  # this is the parent node
        self.nodes = []  # creates an empty list with no child nodes initially
        # self.data = 0  # can be of any value, but just initialized to 0
        self.visit = 1  # when a node is first created, it is counted as visited once
        self.win = 0  # because no play/simulation had been performed yet
        self.loss = 0  # because no play/simulation had been performed yet
        self.puct = 0  # initialized to 0 because game had not started yet

    # this function computes W/N ratio for each node
    def compute_total_win_and_visits(self, total_win=0, visits=0):
        print("compute_total_win_and_visits()")

        if self.win:
            total_win = total_win + 1

        if self.visit:
            visits = visits + 1

        if self.nodes:  # if there is/are child node(s)
            for n in self.nodes:  # traverse down the entire branch for each child node
                n.compute_total_win_and_visits(total_win, visits)

        return total_win, visits  # same order (W/N) as in
        # https://i.imgur.com/uI7NRcT.png inside each node

    # Selection stage of MCTS
    # https://www.reddit.com/r/reinforcementlearning/comments/kfg6qo/selection_phase_of_montecarlo_tree_search/
    def select(self):
        print("select()")
        print("start printing tree for debugging purpose")
        self.print_tree()
        print("finished printing tree")
        # traverse recursively all the way down from the root node
        # to find the path with the highest W/N ratio (this ratio is determined using PUCT formula)
        # and then select that leaf node to do the new child nodes insertion
        leaf = find_best_path(self)  # returns a reference pointer to the desired leaf node
        parent_node = self

        while leaf == -1:
            parent_node = parent_node.nodes
            leaf = find_best_path(parent_node)  # keeps looping in case it is not the leaf yet

        parent_node.insert()  # this leaf node is selected to insert child nodes under it

    # Expansion stage of MCTS
    # Insert Child Nodes for a leaf node
    def insert(self):
        print("insert()")
        # assuming that we are playing tic-tac toe
        # we subtract number of game states already played from the total possible game states
        num_of_possible_game_states = play.TOTAL_NUM__OF_BOXES - play.num_of_play_rounds

        for S in range(num_of_possible_game_states):
            self.nodes.append(Mcts(self))  # inserts child nodes

        # selects randomly just 1 newly added child node and simulate it
        random_child_under_best_parent_node = random.randint(0, num_of_possible_game_states-1)
        self.nodes[random_child_under_best_parent_node].simulate(random_child_under_best_parent_node)

    # Simulation stage of MCTS
    def simulate(self, random_child_under_best_parent_node):
        print("simulate()")
        # best_child_node = find_best_path(self)

        # Instantiates neural network inference coding (play.py) here
        play.mcts_play(is_mcts_in_simulate_stage=1, ongoing_game=play.game_is_on,
                       best_child_node=random_child_under_best_parent_node)
        print("after one round of game")

        if play.game_is_on == 1:  # game not yet finished
            # predicted "intermediate" score during each step of the game,
            # so it is either win (1) or draw (0) or lose (-1)
            print("intermediate out_score = ", play.out_score)

            if play.out_score == 1:
                print("win")
                self.win = 1
                self.loss = 0

            if play.out_score == -1:
                print("lose")
                self.win = 0
                self.loss = 1

            if play.out_score == 0:
                print("draw")
                self.win = 0
                self.loss = 0

            self.backpropagation(self.win, self.loss)

        else:  # game finished
            print(root.print_tree())  # for verifying MCTS logic correctness

    # Backpropagation stage of MCTS
    def backpropagation(self, win, loss):
        print("backpropagation()")
        # traverses upwards to the root node
        # and updates PUCT ratio for each parent nodes
        # computes the PUCT expression Q+U https://slides.com/crem/lc0#/9

        if self.parent == 0:
            num_of_parent_visits = 0
        else:
            num_of_parent_visits = self.parent.visit

        total_win_for_all_child_nodes, num_of_child_visits = self.compute_total_win_and_visits(0, 0)

        self.visit = num_of_child_visits

        # traverses downwards all branches (only for those branches involved in previous play/simulation)
        # and updates PUCT values for all their child nodes
        self.puct = (total_win_for_all_child_nodes / num_of_child_visits) + \
            cfg_puct * np.sqrt(num_of_parent_visits) / (num_of_child_visits + 1)

        if self.parent == 0:  # already reached root node
            self.select()

        else:
            self.parent.visit = self.parent.visit + 1
            if win:
                if self.parent.parent:  # grandparent node (same-coloured player) exists
                    self.parent.parent.win = self.parent.parent.win + 1

            if (win == 0) & (loss == 0):  # tie is between loss (0) and win (1)
                self.parent.win = self.parent.win + 0.5  # parent node (opponent player)

                if self.parent.parent:  # grandparent node (same-coloured player) exists
                    self.parent.parent.win = self.parent.parent.win + 0.5

            self.parent.backpropagation(win, loss)

    # Print the Tree
    def print_tree(self):
        for x in self.nodes:
            print(x.puct)
            if x.nodes:
                self.print_tree()


root = Mcts(0)  # we use parent=0 because this is the head/root node
root.select()
