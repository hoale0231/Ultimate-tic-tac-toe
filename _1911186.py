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
        return evalFunction(cur_state, player), None

    valid_moves = cur_state.get_valid_moves
    if valid_moves == []:
        return evalFunction(cur_state, player), None
    if len(valid_moves) == 1:
        return evalFunction(cur_state, player), valid_moves[0]
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

def evalFunction(cur_state: State, p):
    globalMatrix = cur_state.global_cells.reshape(3,3)
    player = cur_state.player_to_move
    # Check if game is over
    if cur_state.game_result(globalMatrix) is not None:
        if cur_state.game_result(globalMatrix) == player:
            return 100000
        if cur_state.game_result(globalMatrix) == - player:
            return -100000
    
    # Find dangerous blocks
    dangerousBlocks = []
    for ir, row in enumerate(globalMatrix):
        if sum(row) == -2 * player:
            dangerousBlocks.append(ir * 3 + np.where(row == 0)[0][0])
    for ic, col in enumerate(globalMatrix.transpose()):
        if sum(col) == -2 * player:
            dangerousBlocks.append(np.where(col == 0)[0][0] * 3 + ic)
    if sum(globalMatrix.diagonal()) == -2 * player:
        index0 = np.where(globalMatrix.diagonal() == 0)[0][0]
        dangerousBlocks.append(index0 * 3 + index0)
    if sum(globalMatrix[::-1].diagonal()) == -2 * player:
        index0 = np.where(globalMatrix[::-1].diagonal() == 0)[0][0]
        dangerousBlocks.append((2 - index0) * 3 + index0)

    # Calc score
    totalScore = 0
    for ib, block in enumerate(cur_state.blocks):
        dangerous = ib in dangerousBlocks
        winBonus = 0
        score = 0
        flag = np.sum(block == player) > np.sum(block == - player)
        # Check if a block is over
        if cur_state.game_result(block) != None:
            winBonus = 1
        # Score each cell in block
        for row in block:
            for cell in row:
                if cell == player:
                    if winBonus: 
                        score += 10
                    elif dangerous:
                        score += 5
                    
                if cell == - player:
                    if winBonus:
                        score += 10
                    elif dangerous:
                        score -= 5
                    elif flag:
                        score += 0
                    else:
                        score += 3
        totalScore += abs(score) * score 
    return totalScore

        