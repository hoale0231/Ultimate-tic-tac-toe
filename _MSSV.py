import numpy as np
from pathlib import Path
import csv
import os
from state import State, UltimateTTT_Move
from copy import deepcopy
from keras import layers as Kl
from keras import models as Km

class QLearning():
    tag = 'O'
    values = dict()
    alpha = 0.5
    prev_stateStr = ""
    discount_factor = 0.9
    exp_factor = 1
    cntX = 0
    cntO = 0

    @staticmethod
    def make_move(state, winner):
        QLearning.state = state

        if winner is not None:
            new_state = state
            return new_state

        p = np.random.uniform(0, 1)
        if p < QLearning.exp_factor:
            new_state = QLearning.make_optimal_move(state)
        else:
            valid_moves = state.get_valid_moves
            new_state = np.random.choice(valid_moves)

        return new_state

    @staticmethod
    def make_move_and_learn(state, winner):

        QLearning.learn_state(state, winner)

        return QLearning.make_move(state, winner)

    @staticmethod
    def make_optimal_move(state):
        moves = state.get_valid_moves

        if len(moves) == 1:
            return moves[0]

        temp_state_list = []
        v = -float('Inf')

        for move in moves:
            v_temp = []
            tempState = deepcopy(state)
            tempState.act_move(move)
            moves_op = tempState.get_valid_moves

            for move_op in moves_op:
                opState = deepcopy(tempState)
                opState.act_move(move_op)
                v_temp.append(QLearning.calc_value(opState))

            # delets Nones
            v_temp = list(filter(None.__ne__, v_temp))

            if len(v_temp) != 0:
                v_temp = np.min(v_temp)
            else:
                # encourage exploration
                v_temp = 1

            if v_temp > v:
                temp_state_list = [move]
                v = v_temp
            elif v_temp == v:
                temp_state_list.append(move)

        try:
            new_state = np.random.choice(temp_state_list)
        except ValueError:
            print('temp state:', temp_state_list)
            raise Exception('temp state empty')

        return new_state

    @staticmethod
    def state2str(state):
        stateStr = ""
        for block in state.blocks:
            for row in block:
                for cell in row:
                    stateStr += 'X' if cell == 1 else 'O' if cell == -1 else '_'
        if state.previous_move is None:
            return "_"
        return stateStr + str(state.previous_move.x * 3 + state.previous_move.y)

    @staticmethod
    def reward(winner):
        if winner is QLearning.tag:
            return 1
        elif winner is None:
            return 0
        elif winner == 'DRAW':
            return 0.5
        return -1

    @staticmethod
    def learn_state(state, winner):
        stateStr = QLearning.state2str(state)
        if QLearning.tag in stateStr:
            if QLearning.prev_stateStr in QLearning.values.keys():
                prev = QLearning.values[QLearning.prev_stateStr]
            else:
                prev = int(0)

            R = QLearning.reward(winner)
            if R != 0:
                QLearning.values[stateStr] = R

            if stateStr in QLearning.values.keys() and winner is None:
                curr = QLearning.values[stateStr]
            else:
                curr = int(0)

            #temp = prev + QLearning.alpha*(R + curr - prev)
            
            QLearning.values[QLearning.prev_stateStr] = prev + QLearning.alpha*(R + curr - prev)

            # if temp not in {1,-1,0,0.5,-0.5}:
            #     print(QLearning.values[QLearning.prev_stateStr])

        QLearning.prev_stateStr = stateStr

    @staticmethod
    def calc_value(state):
        if state in QLearning.values.keys():
            return QLearning.values[state]
        return None

    @staticmethod
    def load_values():
        s = 'values1' + QLearning.tag + '.csv'
        try:
            value_csv = csv.reader(open(s, 'r'))
            for row in value_csv:
                k, v = row
                QLearning.values[k] = float(v)
        except:
            pass

    @staticmethod
    def save_values():
        s = 'values' + QLearning.tag + '.csv'
        try:
            os.remove(s)
        except:
            pass
        a = csv.writer(open(s, 'a', newline=''))

        for v, k in QLearning.values.items():
            a.writerow([v, k])

#QLearning.load_values()

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
        
class Dlearning():
    tag = State.O
    value_model = None
    alpha = 0.7

    @staticmethod
    def make_move(state, winner):
        Dlearning.state = state

        if winner is not None:
            new_state = state
            return new_state

        p = np.random.uniform(0, 1)
        if p < Dlearning.exp_factor:
            new_state = Dlearning.make_optimal_move(state)
        else:
            valid_moves = state.get_valid_moves
            new_state = np.random.choice(valid_moves)

        return new_state

    @staticmethod
    def make_move_and_learn(state, winner):

        Dlearning.learn_state(state, winner)

        return Dlearning.make_move(state, winner)

    @staticmethod
    def make_optimal_move(state):
        moves = state.get_valid_moves

        if len(moves) == 1:
            return moves[0]

        temp_state_list = []
        v = -float('Inf')

        for move in moves:
            v_temp = []
            tempState = deepcopy(state)
            tempState.act_move(move)
            moves_op = tempState.get_valid_moves

            for move_op in moves_op:
                opState = deepcopy(tempState)
                opState.act_move(move_op)
                v_temp.append(Dlearning.calc_value(opState))

            # delets Nones
            v_temp = list(filter(None.__ne__, v_temp))

            if len(v_temp) != 0:
                v_temp = np.min(v_temp)
            else:
                # encourage exploration
                v_temp = 1

            if v_temp > v:
                temp_state_list = [move]
                v = v_temp
            elif v_temp == v:
                temp_state_list.append(move)

        try:
            new_state = np.random.choice(temp_state_list)
        except ValueError:
            print('temp state:', temp_state_list)
            raise Exception('temp state empty')

        return new_state

    @staticmethod
    def reward(winner):
        if winner is Dlearning.tag:
            return 1
        elif winner is None:
            return 0
        elif winner == 'DRAW':
            return 0.5
        return -1

    @staticmethod
    def state2array(state):

        num_state = []
        for s in state:
            if s == 'X':
                num_state.append(1)
            elif s == 'O':
                num_state.append(-1)
            else:
                num_state.append(0)
        num_state = np.array([num_state])
        return num_state

    @staticmethod
    def learn_state(state, winner):

        target = Dlearning.calc_target(state, winner)

        Dlearning.train_model(target, 10)

        Dlearning.prev_state = state

    @staticmethod
    def load_model():
        s = 'model_values' + Dlearning.tag + '.h5'
        model_file = Path(s)
        if model_file.is_file():
            model = Km.load_model(s)
            print('load model: ' + s)
        else:
            print('new model')
            model = Km.Sequential()
            model.add(Kl.Dense(18, activation='relu', input_dim=9))
            model.add(Kl.Dense(18, activation='relu'))
            model.add(Kl.Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

        model.summary()
        Dlearning.value_model = model

    @staticmethod
    def calc_value(state):
        return Dlearning.value_model.predict(Dlearning.state2array(state))

    @staticmethod
    def calc_target(state, winner):

        if Dlearning.tag in state:

            v_s = Dlearning.calc_value(Dlearning.prev_state)

            R = Dlearning.reward(winner)

            if winner is None:
                v_s_tag = Dlearning.calc_value(state)
            else:
                v_s_tag = 0

            target = np.array(v_s + Dlearning.alpha * (R + 0.8 * v_s_tag - v_s))

            return target

    @staticmethod
    def train_model(target, epochs):

        X_train = Dlearning.state2array(Dlearning.prev_state)

        if target is not None:
            Dlearning.value_model.fit(X_train, target, epochs=epochs, verbose=0)

    @staticmethod
    def save_values():
        s = 'model_values' + Dlearning.tag + '.h5'
        try:
            os.remove(s)
        except:
            pass
        Dlearning.value_model.save(s)

#Dlearning.load_model()

def select_move(cur_state: State, remain_time, winner = None):
    valid_moves = cur_state.get_valid_moves 
    if len(valid_moves) != 0:
        if cur_state.player_to_move == State.X:
            return MoveFirst.selectMove(cur_state)
        else:
            return alphabeta(cur_state, 4, float('-inf'), float('inf'), State.O)
    return None

def alphabeta(cur_state, depth, alpha, beta, player):
    if depth == 0:
        return evalFunction(cur_state), None

    valid_moves = cur_state.get_valid_moves()
    if valid_moves == []:
        return evalFunction(cur_state), None
    if len(valid_moves) == 1:
        return evalFunction(cur_state), valid_moves[0]
    
    if player == 1:
        bestMove = None
        value = float('-inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = max(value, alphabeta(newState, depth - 1, alpha, beta, -player)[0])
            if value >= beta:
                break
            if value > alpha:
                alpha = value
                bestMove = move
        return alpha, bestMove
    else:
        bestMove = None
        value = float('inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = min(value, alphabeta(newState, depth - 1, alpha, beta, -player)[0])
            if value <= alpha:
                break
            if value < beta:
                beta = value
                bestMove = move
        return beta, bestMove

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
