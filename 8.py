import numpy as np

from state import UltimateTTT_Move
from state import State_2

def is_full_block(box):
    for row in box:
        for cell in row:
            if cell == 0:
                return False

    return True

def caculate_value(cur_state):
    score = 0
    for i in range(0, 9):
        box = cur_state.blocks[i]
        score += eval_box(box,cur_state.player_to_move)
    return score

def eval_box(box,player):
    score = 0

     #Score for each row
    for row_index in range(0, 3):
        row = []
        for col_index in range(0, 3):
            row.append(box[row_index][col_index])

        score += count_score(row, player)

    #Score for each column
    for col_index in range(0, 3):
        col = []
        for row_index in range(0, 3):
            col.append(box[row_index][col_index])

        score += count_score(col, player)

    #Score for each diagonal
    diags = []
    for indx in range(0, 3):
        diags.append(box[indx][indx])

    score += count_score(diags, player)

    diags_2 = []
    for indx, rev_indx in enumerate(reversed(range(3))):
        diags_2.append(box[indx][rev_indx])
    score += count_score(diags_2, player)

    if is_full_block(box):
        score += 1

    return score

def count_score(array, player):
    opp_player = -player
    score = 0

    if array.count(player) == 3:
        score += 100

    elif array.count(player) == 2:
        score += 50

    elif array.count(player) == 1:
        score += 20

    if array.count(opp_player) == 3:
        score -= 100

    elif array.count(opp_player) == 2:
        score -= 50

    if array.count(player) == 1 and array.count(opp_player) == 2:
        score += 10

    return score

def check_full_block_with_move(cur_state, move):
    index_block_check = move.x * 3 + move.y
    return is_full_block(cur_state.blocks[index_block_check])

def next_state_with_action(cur_state, move):
    next_state = State_2(cur_state)
    if cur_state.free_move == True:
        next_state.free_move = True
    next_state.act_move(move)
    return next_state

def minimax(cur_state, depth, alpha, beta, maxmizingPlayer):
    if depth == 0 or cur_state.game_over:
        return caculate_value(cur_state), None
    
    valid_moves = cur_state.get_valid_moves
    next_move = None
    
    if maxmizingPlayer:
        maxValue = float('-inf')
        if len(valid_moves) == 1:
            next_move = valid_moves[0]
            next_state = next_state_with_action(cur_state, next_move)
            value = minimax(next_state, depth - 1, alpha, beta, False)[0]
            if value > maxValue:
                maxValue = value
            alpha = max(value, alpha)
            return maxValue, next_move
        for _move in valid_moves:
            next_state = next_state_with_action(cur_state, _move)
            if next_state == None:
                continue
            value = minimax(next_state, depth - 1, alpha, beta, False)[0]
            if value > maxValue:
                next_move = _move
                maxValue = value
            alpha = max(value, alpha)
            if beta <= alpha:
                break
        return maxValue, next_move

    else:
        minValue = float('+inf')
        if len(valid_moves) == 1:
            next_move = valid_moves[0]
            next_state = next_state_with_action(cur_state, next_move)
            value = minimax(next_state, depth - 1, alpha, beta, True)[0]
            if value < minValue:
                minValue = value
            beta = min(value, beta)
            return minValue, valid_moves[0]
        for _move in valid_moves:
            next_state = next_state_with_action(cur_state, _move)
            if next_state == None:
                continue
            value = minimax(next_state, depth - 1, alpha, beta, True)[0]
            if value < minValue:
                next_move = _move
                minValue = value
            beta = min(value, beta)
            if beta <= alpha:
                break
        return minValue, next_move

def select_move(cur_state, remain_time):
    if cur_state.player_to_move == 1:
        move = minimax(cur_state, 3, float('-inf'), float('+inf'), True)[1]
        return move
                
    elif cur_state.player_to_move == -1:
        move = minimax(cur_state, 3, float('-inf'), float('+inf'), False)[1]
        return move