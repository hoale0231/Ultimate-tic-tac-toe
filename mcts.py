#!/usr/bin/env python3

from copy import deepcopy
from time import time
from numpy import log, sqrt, random
import numpy as np
from state import State, State_2, UltimateTTT_Move

symbol = int

def two_consecutive(board, weight):
    """
    Reward for winning two global cell in a row or diagonally
    """

    total_score = 0
    row_sum = np.sum(board, 1)
    col_sum = np.sum(board, 0)
    diag_sum_topleft = board.trace()
    diag_sum_topright = board[::-1].trace()
    for sum in list(row_sum) + list(col_sum) + [diag_sum_topleft, diag_sum_topright]:
        if abs(sum) == 2:
            total_score += sum * weight

    return total_score

def evaluate_board(state: State_2):
    valid_moves = state.get_valid_moves
    successors: list[State_2] = []
    scores = []
    for move in valid_moves:
        new_state = deepcopy(state)
        new_state.act_move(move)
        successors.append(new_state)
        
    for successor in successors:
        score = 0
        if successor.game_over:
            score = np.inf
            continue
        
        for cell in state.global_cells:
            # for each cell wins, score += 200
            score += cell*200

        score += state.global_cells[4]*150

        for cell in state.blocks:
            score += cell[1][1]*20
            score += two_consecutive(cell, 20)
        
        score += two_consecutive(state.global_cells.reshape(3, 3), 250)
        
        scores.append(score)
    
        i = scores.index(max(scores))
        return successors[i].previous_move

class MCTSNode:
    def __init__(self, parent, state: State) -> None:
        self.parent = parent
        self.state = state
        self.children: list[MCTSNode] = []
        self.score = self.total_tries = 0


    def explore(self) -> None:
        valid_moves = self.state.get_valid_moves
        for move in valid_moves:
            new_state = deepcopy(self.state)
            new_state.act_move(move)

            new_node = MCTSNode(self, new_state)
            self.children.append(new_node)


    def back_propagate(self, result: int) -> None:
        self.total_tries += 1
        self.score += result

        # if this is a non-root node
        if self.parent:
            self.parent.back_propagate(result)

    @property
    def exploration_term(self) -> float:
        return sqrt(log(self.parent.total_tries or 1)/(self.total_tries or 1))


    @property
    def exploitation_term(self) -> float:
        # if not self.total_tries:
        #     return 1
        return self.score/(self.total_tries or 1)

    def __repr__(self) -> str:
        return f'score: {self.score}, total tries: {self.total_tries}'

class MCTS:
    def __init__(self, turn, C=sqrt(2), time_limit=2) -> None:
        self.turn = turn
        self.C = C
        self.time_limit = time_limit


    def simulate(self, state: State) -> int:
        valid_moves = state.get_valid_moves
        if not state.game_over and valid_moves:
            # best_state = None
            # best_score = -100000000
            # for move in valid_moves:
            #     new_state = deepcopy(state)
            #     new_state.act_move(move)
            #     score = calculate_score(new_state)
            #     if score > best_score:
            #         best_score = score
            #         best_state = new_state
            # move = random.choice(valid_moves)
            # move = evaluate_board(state)
            # if not move:
            move = random.choice(valid_moves)
            state.act_move(move)

            return self.simulate(state)

        if not valid_moves:
            return 0

        if state.player_to_move == self.turn:
            return -1

        return 1
        # return state.player_to_move


    def choose_best_child(self, cur_node: MCTSNode, turn: symbol):
        # if is terminal node or the node has no children
        if cur_node.state.game_over or not cur_node.children:
            return cur_node

        best_child: MCTSNode
        if turn == self.turn:
            best_child = sorted(cur_node.children, key=lambda child: child.exploitation_term + self.C*child.exploration_term, reverse=True)[0]
        else:
            best_child = sorted(cur_node.children, key=lambda child: -child.exploitation_term + self.C*child.exploration_term, reverse=True)[0]

        return self.choose_best_child(best_child, self.turn*(-1))


    def get_potential_move(self, state: State_2) -> UltimateTTT_Move:
        root_node = MCTSNode(None, deepcopy(state))
        start_time = time()
        potential_child = root_node
        while time() - start_time < self.time_limit:
            potential_child = self.choose_best_child(root_node, self.turn)

            if potential_child.total_tries > 0:
                potential_child.explore()
            else: # First simulation
                result = self.simulate(deepcopy(potential_child.state))
                potential_child.back_propagate(result)

        nodes = sorted(root_node.children, key=lambda child: child.exploitation_term, reverse=True)
        for node in nodes:
            if state.is_valid_move(node.state.previous_move):
                return node.state.previous_move

        return evaluate_board(state)
