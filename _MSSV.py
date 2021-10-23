import numpy as np
import state


def evaluate(move):
    return 0


def alphabeta(cur_state, depth, alpha, beta, player):
    if depth == 0:
        return evaluate(cur_state), None

    valid_moves = cur_state.get_valid_moves()
    if valid_moves == []:
        return evaluate(cur_state), None
    elif len(valid_moves) == 1:
        return evaluate(cur_state), valid_moves[0]
    else:
        if player == 1:
            value = float("-inf")
            for move in valid_moves:
                # cur_state chuyen trang thai??
                value = max(value, alphabeta(
                    move, depth - 1, alpha, beta, -player)[0])
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, move
        else:
            value = float("-inf")
            for move in valid_moves:
                value = min(value, alphabeta(
                    move, depth - 1, alpha, beta, -player)[0])
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, move


def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves

    if len(valid_moves) != 0:
        return np.random.choice(valid_moves)
    return None

    # value, move = alphabeta(cur_state, 4, float("-inf"), float("inf"), 1)
    # return move
