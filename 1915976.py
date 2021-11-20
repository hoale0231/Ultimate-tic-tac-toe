import numpy as np
from numpy.core.records import array
from state import State, State_2, UltimateTTT_Move
from copy import deepcopy

def select_move(cur_state: State_2, remain_time, winner=None):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        if cur_state.player_to_move == State.X:
            return alphabeta(cur_state, 2, float('-inf'), float('inf'), 1, 1)[1]
            # return MoveFirst.selectMove(cur_state)
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

def computeLine(arr):
    countX, countO = np.count_nonzero(arr == 1), np.count_nonzero(arr == -1)
    count_ = 3 - (countO + countX) 
    score = 0

    if (count_ > 0 or (countX > 0 == countO > 0)):
        score = 0
    elif (countX > 0):
        score = pow(10, countX)
    else:
        score = -pow(10, countO)
    return score

def add(arr):
    result = 0   

    result += computeLine(arr[0]) + computeLine(arr[1]) + computeLine(arr[2])
    result += computeLine(arr[:,0]) + computeLine(arr[:,1]) + computeLine(arr[:,2])
    result += computeLine([arr[i][i] for i in range(3)]) + computeLine([arr[i][2-i] for i in range(3)])

    # print(result)
    return result

def evalFunction(cur_state: State_2, flag):
    finalScore = 0

    # finalScore += add(new_shape, flag)
    Board = cur_state.blocks
    for block in cur_state.blocks:
        finalScore += add(block)

    finalScore -= (computeLine([cur_state.game_result(Board[0]), cur_state.game_result(Board[1]), cur_state.game_result(Board[2])]) 
                + computeLine([cur_state.game_result(Board[3]), cur_state.game_result(Board[4]), cur_state.game_result(Board[5])]) 
                + computeLine([cur_state.game_result(Board[6]), cur_state.game_result(Board[7]), cur_state.game_result(Board[8])]))
    finalScore -= (computeLine([cur_state.game_result(Board[0]), cur_state.game_result(Board[3]), cur_state.game_result(Board[6])]) 
                + computeLine([cur_state.game_result(Board[1]), cur_state.game_result(Board[4]), cur_state.game_result(Board[7])]) 
                + computeLine([cur_state.game_result(Board[2]), cur_state.game_result(Board[5]), cur_state.game_result(Board[8])]))
    finalScore -= (computeLine([cur_state.game_result(Board[0]), cur_state.game_result(Board[4]), cur_state.game_result(Board[8])])
                + computeLine([cur_state.game_result(Board[2]), cur_state.game_result(Board[4]), cur_state.game_result(Board[6])])) 

    # print(flag, finalScore)

    return finalScore*flag
