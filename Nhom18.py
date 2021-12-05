from numpy.lib.function_base import piecewise, place
from state import State_2
from copy import deepcopy
import numpy as np

X = 1
O = -1
aiPlayer = X

def select_move(cur_state: State_2, remain_time, winner = None):
    global aiPlayer 
    aiPlayer = cur_state.player_to_move
    valid_moves = cur_state.get_valid_moves 
    if len(valid_moves) != 0:
        return alphabeta(cur_state, 3, float('-inf'), float('inf'), 1)[1]
    return None

def alphabeta(cur_state: State_2, depth, alpha, beta, player):
    if depth == 0:
        return evalFunction(cur_state), None

    valid_moves = cur_state.get_valid_moves
    if valid_moves == []:
        return evalFunction(cur_state), None
    if len(valid_moves) == 1:
        return evalFunction(cur_state), valid_moves[0]
    #if depth == 2:
        #print('====================')
    if player == 1:
        bestMove = []
        bestVal = float('-inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = alphabeta(newState, depth - 1, alpha, beta, -player)[0]
            #if depth == 2:
                #print(value)
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
            value = alphabeta(newState, depth - 1, alpha, beta, -player)[0]
            #if(depth == 2):
                 #print(value)
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

def evalFunction(state: State_2):
    eval = 0    
    blockresults = np.array([state.game_result(block) for block in state.blocks]).reshape(3, 3)
    for line in list(blockresults) + list(blockresults[::-1]) + [np.diag(blockresults), np.diag(blockresults[::-1])]:
        if np.any(line == 0):
            continue
        numX = np.sum(line == 1)
        numY = np.sum(line == -1)

        if (numX > 0) == (numY > 0):
            continue

        if numX > 0:
            eval += 10 ** (numX + 1)
        else:
            eval -= 10 ** (numY + 1)
              
    for block in state.blocks:
        for line in list(block) + list(block[::-1]) + [np.diag(block), np.diag(block[::-1])]:
            numX = np.sum(line == 1)
            numY = np.sum(line == -1)
            
            if (numX > 0) == (numY > 0):
                continue
            
            if numX > 0:
                eval += 10 ** (numX - 1)
            else:
                eval -= 10 ** (numY - 1)
    #print(eval * aiPlayer)
    return eval * aiPlayer
        
        
        
    


        