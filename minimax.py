from state import State, UltimateTTT_Move
from copy import deepcopy
import numpy as np

class MoveFirst:
    phase = 1
    target = -1
    oppostie = -1
    
    @staticmethod
    def findTarget(state: State):
        MoveFirst.target = [np.all(block == 0) for block in state.blocks].index(True)
        MoveFirst.oppostie = list(range(9))[-MoveFirst.target-1]

    @staticmethod
    def selectMove(state: State):
        if np.all(state.blocks == 0):
            MoveFirst.phase = 1
        if MoveFirst.phase == 1:
            MoveFirst.phase = 2
            return UltimateTTT_Move(4, 1, 1, 1)
        if MoveFirst.phase == 2:
            if sum([np.all(block == 0) for block in state.blocks]) > 1:
                for move in state.get_valid_moves:
                    if move.x == 1 and move.y == 1:
                        return move
            MoveFirst.findTarget(state)
            MoveFirst.phase = 3
        if MoveFirst.phase == 3:
            if state.previous_move.x * 3 + state.previous_move.y == MoveFirst.oppostie or (state.previous_move.x == 1 and state.previous_move.y == 1):
                move = UltimateTTT_Move(MoveFirst.oppostie, MoveFirst.target // 3, MoveFirst.target % 3, 1)
                if state.is_valid_move(move):
                    return move
                return UltimateTTT_Move(MoveFirst.oppostie, MoveFirst.oppostie // 3, MoveFirst.oppostie % 3, 1)

            for move in state.get_valid_moves:
                if move.x == MoveFirst.target // 3 and move.y == MoveFirst.target % 3:
                    return move
            
            for move in state.get_valid_moves:
                if move.x == MoveFirst.oppostie // 3 and move.y == MoveFirst.oppostie % 3:
                    return move
        
def select_move(cur_state: State, remain_time, winner = None):
    valid_moves = cur_state.get_valid_moves 
    if len(valid_moves) != 0:
        if cur_state.player_to_move == State.X:
            return MoveFirst.selectMove(cur_state)
        else:
            return alphabeta(cur_state, 2, float('-inf'), float('inf'), 1)[1]
    return None

def alphabeta(cur_state: State, depth, alpha, beta, player):
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

def add(arr, number):
    result = 0

    row_sum = np.sum(arr, 1)
    col_sum = np.sum(arr, 0)
    diag_sum_topleft = arr.trace()
    diag_sum_topright = arr[::-1].trace()

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

    if (0 not in arr[0]):
        if (row_sum[0] == -1):  result-=number
        else:   result+=number
    if (0 not in arr[1]):
        if (row_sum[1] == -1):  result-=number
        else:   result+=number
    if (0 not in arr[2]):
        if (row_sum[2] == -1):  result-=number
        else:   result+=number
    if (arr[0][0] != 0 and arr[1][0] != 0 and arr[2][0] != 0):
        if (col_sum[0] == -1):  result-=number
        else:   result+=number
    if (arr[0][1] != 0 and arr[1][1] != 0 and arr[2][1] != 0):
        if (col_sum[1] == -1):  result-=number
        else:   result+=number
    if (arr[0][2] != 0 and arr[1][2] != 0 and arr[2][2] != 0):
        if (col_sum[2] == -1):  result-=number
        else:   result+=number
    if (arr[0][0] != 0 and arr[1][1] != 0 and arr[2][2] != 0):
        if (diag_sum_topleft == -1):    result-=number
        else:   result+=number
    if (arr[0][2] != 0 and arr[1][1] != 0 and arr[2][0] != 0):
        if (diag_sum_topright == -1):    result-=number
        else:   result+=number

    MatrixScore = np.array([[3,2,3],
                            [2,4,2],
                            [3,2,3]])

    result -= np.sum(arr*MatrixScore)*5
    return result

def evalFunction(cur_state: State):
    if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) is not None:
        if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == cur_state.player_to_move:
            return 100000
        if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == - cur_state.player_to_move:
            return -100000
    new_shape = cur_state.global_cells.reshape(3, 3)
    finalScore = 0

    for block in cur_state.blocks:
        # Suppose player is O, competitor is X
        if cur_state.game_result(block) == cur_state.O:
            finalScore += 200
        elif cur_state.game_result(block) == cur_state.X:
            finalScore -= 150
        else:    
            finalScore += add(block, 20)
    return finalScore
