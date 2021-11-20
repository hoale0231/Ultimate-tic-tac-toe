import numpy as np
from state import State, State_2, UltimateTTT_Move
from copy import deepcopy


def select_move(cur_state: State_2, remain_time, winner=None):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        if cur_state.player_to_move == State.X:
            return alphabeta(cur_state, 2, float('-inf'), float('inf'), 1, 1)[1]
        else:
            return alphabeta(cur_state, 2, float('-inf'), float('inf'), 1, -1)[1]

    return None


def alphabeta(cur_state: State_2, depth, alpha, beta, player, flag):
    if depth == 0:
        return evalFunction(cur_state, flag), None

    valid_moves = cur_state.get_valid_moves
    if valid_moves == []:
        return evalFunction(cur_state, flag), None
    if len(valid_moves) == 1:
        return evalFunction(cur_state, flag), valid_moves[0]
    if player == 1:
        bestMove = []
        bestVal = float('-inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = alphabeta(newState, depth - 1, alpha,
                              beta, -player, flag)[0]
            # if depth == 2:
            #     print(value)
            if value > bestVal:
                bestVal = value
                bestMove = [move]
            if value == bestVal:
                bestMove.append(move)
            if bestVal >= beta:
                break
            if bestVal > alpha:
                alpha = bestVal
        return bestVal, np.random.choice(bestMove)
    else:
        bestMove = None
        bestVal = float('inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = alphabeta(newState, depth - 1, alpha,
                              beta, -player, flag)[0]
            # if(depth == 2):
            #     print(value)
            if value < bestVal:
                bestVal = value
                bestMove = [move]
            if value == bestVal:
                bestMove.append(move)
            if alpha >= bestVal:
                break
            if bestVal < beta:
                beta = bestVal
        return bestVal, np.random.choice(bestMove)


def add(cur_state: State_2, arr, number, flag):
    result = 0
    row_sum = np.sum(arr, 1)
    col_sum = np.sum(arr, 0)
    diag_sum_topleft = arr.trace()
    diag_sum_topright = arr[::-1].trace()
    if flag == -1:
        number_of_row = np.count_nonzero(row_sum == 2)
        result -= number_of_row * number

        number_of_col = np.count_nonzero(col_sum == 2)
        result -= number_of_col * number
        if diag_sum_topleft == 2:
            result -= number
        if diag_sum_topright == 2:
            result -= number

        number_of_row = np.count_nonzero(row_sum == -2)
        result += number_of_row * number

        number_of_col = np.count_nonzero(col_sum == -2)
        result += number_of_col * number
        if diag_sum_topleft == -2:
            result += number
        if diag_sum_topright == -2:
            result += number
    elif flag == 1:
        number_of_row = np.count_nonzero(row_sum == 2)
        result += number_of_row * number

        number_of_col = np.count_nonzero(col_sum == 2)
        result += number_of_col * number
        if diag_sum_topleft == 2:
            result += number
        if diag_sum_topright == 2:
            result += number

        number_of_row = np.count_nonzero(row_sum == -2)
        result -= number_of_row * number

        number_of_col = np.count_nonzero(col_sum == -2)
        result -= number_of_col * number
        if diag_sum_topleft == -2:
            result -= number
        if diag_sum_topright == -2:
            result -= number

    return result


def evalFunction(cur_state: State_2, flag):
    if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) is not None:
        if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == cur_state.player_to_move:
            return 100000
        if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == - cur_state.player_to_move:
            return -100000
    new_shape = cur_state.global_cells.reshape(3, 3)
    finalScore = 0

    finalScore += add(cur_state, new_shape, 100, flag)

    for block in cur_state.blocks:
        # Suppose player is O, competitor is X
        if cur_state.game_result(block) == cur_state.O:
            finalScore += 30
        if cur_state.game_result(block) == cur_state.X:
            finalScore -= 30
        finalScore += add(cur_state, block, 10, flag)

    return finalScore
