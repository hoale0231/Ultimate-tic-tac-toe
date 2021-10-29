
from collections import deque
import numpy as np
from state import State
from copy import deepcopy

def select_move(cur_state: State, remain_time):
    valid_moves = cur_state.get_valid_moves 
    if len(valid_moves) != 0:
        return alphabeta(cur_state, 4, float('-inf'), float('inf'), cur_state.player_to_move)[1]
    return None

def alphabeta(cur_state: State, depth, alpha, beta, player):
    if depth == 0:
        return evalFunction(cur_state), None

    valid_moves = cur_state.get_valid_moves
    if valid_moves == []:
        return evalFunction(cur_state), None
    if len(valid_moves) == 1:
        return evalFunction(cur_state), valid_moves[0]

    if player == 1:
        bestMove = None
        bestVal = float('-inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            bestVal = alphabeta(newState, depth - 1, alpha, beta, -player)[0]
            if bestVal >= beta:
                break
            if bestVal > alpha:
                alpha = bestVal
                bestMove = move
        return bestVal, bestMove
    else:
        bestMove = None
        bestVal = float('inf')
        for move in valid_moves:    
            newState = deepcopy(cur_state)
            newState.act_move(move)
            bestVal = alphabeta(newState, depth - 1, alpha, beta, -player)[0]
            if alpha >= bestVal:
                break
            if bestVal < beta:
                beta = bestVal
                bestMove = move
        return bestVal, bestMove

def evalFunction(cur_state):

    index_local_board = cur_state.previous_move.x * 3 + cur_state.previous_move.y
    local_board = cur_state.blocks[index_local_board].reshape(9)

    row_sum = np.sum(cur_state.blocks[index_local_board],1)
    col_sum = np.sum(cur_state.blocks[index_local_board],0)
    diagional_Left = local_board[0] + local_board[4] + local_board[8]
    diagional_Right = local_board[2] + local_board[4] + local_board[6]

    FirstWin = any(row_sum == 3) + any(col_sum == 3)
    FirstWin += (diagional_Left == 3) + (diagional_Right == 3)

    if FirstWin:
        return cur_state.X * 100
    
    secondWin = any(row_sum == -3) + any(col_sum == -3)
    secondWin += (diagional_Left == -3) + (diagional_Right == -3)

    if secondWin:
        return cur_state.O*100

    X2 = 0
    X1 = 0

    for i in range(0,2):
        if (row_sum[i] == 1):
            if (local_board[3*i+1] != -1 and local_board[3*i + 2] != -1 and local_board[3*i] != -1):
                X1 += 1
        if (col_sum[i] == 1):
            if (local_board[i+3] != -1 and local_board[i] != -1 and local_board[i + 6] != -1):
                X1 += 1

    if (diagional_Left == 1 and local_board[0] != -1 and local_board[4] != -1 and local_board[8] != -1):
        X1 += 1
    if (diagional_Right == 1 and local_board[2] != -1 and local_board[4] != -1 and local_board[6] != -1):
        X1 += 1


    O2 = 0
    O1 = 0

    for i in range(0,2):
        if (row_sum[i] == -1):
            if (local_board[3*i+1] != 1 and local_board[3*i + 2] != 1 and local_board[3*i] != 1):
                O1 += 1
        if (col_sum[i] == -1):
            if (local_board[i+3] != 1 and local_board[i] != 1 and local_board[i + 6] != 1):
                O1 += 1

    if (diagional_Left == -1 and local_board[0] != 1 and local_board[4] != 1 and local_board[8] != 1):
        O1 += 1
    if (diagional_Right == -1 and local_board[2] != 1 and local_board[4] != 1 and local_board[6] != 1):
        O1 += 1

    X2 += np.count_nonzero(row_sum == 2) + np.count_nonzero(col_sum == 2) + diagional_Left == 2 + diagional_Right == 2 
    O2 += np.count_nonzero(row_sum == -2) + np.count_nonzero(row_sum == -2) + diagional_Left == -2 + diagional_Right == -2
  
     
    return (3*X2 + X1) - (3*O2 + O1) 
