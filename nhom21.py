from typing import Optional
# import numpy as np
from state import State, UltimateTTT_Move
from copy import deepcopy
from mcts import MCTS
#from mctsv2 import select_potential_move

INF = float('inf')
X = 1
O = -1

def evaluate(state: State):
    if state.game_over:
        if state.player_to_move == O:
            return 100
        else:
            return -100
    return state.count_O - state.count_X

def ABpruning(
        state: State,
        move: Optional[UltimateTTT_Move],
        depth: int,
        turn: int,
        alpha: float,
        beta: float):
    # Get all possible moves
    valid_moves = state.get_valid_moves

    # If the state tree has reached the max depth
    # or if there are no possible moves
    # then backtrack
    if depth == 0 or not valid_moves:
        return move, evaluate(state)

    # Create all possible branchs/successor state
    successors: list[tuple[Optional[UltimateTTT_Move], State]] = []
    for move in valid_moves:
        succ_state = deepcopy(state)
        # succ_state = State_2(state)
        succ_state.act_move(move)
        successors.append((move, succ_state))

    # Iterate through all successors and evaluate
    best_eval = -INF
    best_move = None
    if turn == O:
        for move, successor in successors:
            _, eval = ABpruning(successor, move, depth - 1, X, alpha, beta)
            best_eval = max(best_eval, eval)
            alpha = max(alpha, best_eval)
            if beta <= alpha:
                break
            if best_eval == eval:
                best_move = move
    else:
        best_eval *= -1
        for move, successor in successors:
            _, eval = ABpruning(successor, move, depth - 1, O, alpha, beta)
            best_eval = min(best_eval, eval)
            beta = min(beta, best_eval)
            if beta <= alpha:
                break
            if best_eval == eval:
                best_move = move

    return best_move, best_eval

def select_move(cur_state: State, remain_time: float) -> Optional[UltimateTTT_Move]:
    # best_move, _ = ABpruning(cur_state, None, 6, cur_state.player_to_move, -INF, INF)
    mcts = MCTS(turn=cur_state.player_to_move, time_limit=2.2)
    # print(cur_state)
    return mcts.get_potential_move(cur_state)
