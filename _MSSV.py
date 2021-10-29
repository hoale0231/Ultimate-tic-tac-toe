from time import sleep
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
        return stateStr

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
            
            QLearning.values[QLearning.prev_stateStr] = prev + QLearning.alpha*(R + QLearning.discount_factor*curr - prev)
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
    if depth == 2:
        print('====================')
    if player == 1:
        bestMove = []
        bestVal = float('-inf')
        for move in valid_moves:
            newState = deepcopy(cur_state)
            newState.act_move(move)
            value = alphabeta(newState, depth - 1, alpha, beta, -player)[0]
            if depth == 2:
                print(value)
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
            if(depth == 2):
                 print(value)
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

def evalFunction(cur_state: State):
    if cur_state.game_result(cur_state.global_cells.reshape(3,3)) is not None:
        if cur_state.game_result(cur_state.global_cells.reshape(3,3)) == cur_state.player_to_move:
            return 100000
        if cur_state.game_result(cur_state.global_cells.reshape(3,3)) == - cur_state.player_to_move:
            return -100000
    #print(cur_state.player_to_move)
    # dangerous = []
    # for row in cur_state.global_cells.reshape(3,3):
    #     if sum(row == )    
    # row_sum = np.sum(cur_state.global_cells, 1)
    # col_sum = np.sum(cur_state.global_cells, 0)
    # diag_sum_topleft = cur_state.global_cells.trace()
    # diag_sum_topright = cur_state.global_cells[::-1].trace()
    
    # player_one_wins = any(row_sum == 2) + any(col_sum == 2)
    # player_one_wins += (diag_sum_topleft == 2) + (diag_sum_topright == 2)
    
    total = 0
    for block in cur_state.blocks:
        winBonus = 1
        score = 0
        if cur_state.game_result(block) != None:
            winBonus = 10
        for row in block:
            for cell in row:
                if cell == cur_state.player_to_move:
                    if winBonus > 1: score += 3
                    else: score -= 1
                if cell == - cur_state.player_to_move:
                    score += 3
        total += score * score * winBonus
    return total

    # value = 0
    # for block in cur_state.blocks:
        