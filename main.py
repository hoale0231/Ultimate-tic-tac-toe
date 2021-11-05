from state import State, State_2
import time
from importlib import import_module

  
def main(player_X, player_O, rule = 1):
    dict_player = {1: 'X', -1: 'O'}
    if rule == 1:
        cur_state = State()
    else:
        cur_state = State_2()
    turn = 1    

    limit = 81
    remain_time_X = 120
    remain_time_O = 120
    
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    
    
    while turn <= limit:
        #print("turn:", turn, end='\n\n')
        if cur_state.game_over:
            #print("winner:", dict_player[cur_state.player_to_move * -1])
            return dict_player[cur_state.player_to_move * -1]
            break
        
        start_time = time.time()
        if cur_state.player_to_move == 1:
            new_move = player_1.select_move(cur_state, remain_time_X)
            elapsed_time = time.time() - start_time
            remain_time_X -= elapsed_time
        else:
            new_move = player_2.select_move(cur_state, remain_time_O)
            elapsed_time = time.time() - start_time
            remain_time_O -= elapsed_time
            
        if new_move == None:
            break
        
        if remain_time_X < 0 or remain_time_O < 0:
            #print("out of time")
            #print("winner:", dict_player[cur_state.player_to_move * -1])
            return dict_player[cur_state.player_to_move * -1]
            break
                
        if elapsed_time > 10.0:
            #print("elapsed time:", elapsed_time)
            #print("winner: ", dict_player[cur_state.player_to_move * -1])
            return dict_player[cur_state.player_to_move * -1]
            break
        
        cur_state.act_move(new_move)
        #print(cur_state)
        
        turn += 1
    #print('DRAW')
    if cur_state.count_X > cur_state.count_O:
        return 'X'
    elif cur_state.count_O > cur_state.count_X:
        return 'O'
    else:
        return 'DRAW'

    #print("X:", cur_state.count_X)
    #print("O:", cur_state.count_O)

# cntX = 0
# cntO = 0
# cntDraw = 0
# for i in range(100):
#     if i % 10 == 0:
#         print(f'Done: {i}%')
#     rs = main('_mssv_quynh', 'random_agent')
#     if rs == 'O':
#         cntO += 1
#     if rs == 'X':
#         cntX += 1
#     if rs == 'DRAW':
#         cntDraw += 1

# print(f'X Win: {cntX}\n O Win: {cntO}\n Draw:{cntDraw}') 

cntX = 0
cntO = 0
cntDraw = 0
for i in range(100):
    if i % 10 == 0:
        print(f'Done: {i}%')

    rs = main('_1915976', '_1911186_1915976_1914900')
    if rs == 'O':
        cntO += 1
    if rs == 'X':
        cntX += 1
    if rs == 'DRAW':
        cntDraw += 1

print(f'X Win: {cntX}\n O Win: {cntO}\n Draw:{cntDraw}') 
'_1911186_1915976_1914900'
'_mssv_quynh'
'_1911186'
'random_agent'
'_1915976'