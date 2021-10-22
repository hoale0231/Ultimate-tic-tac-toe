import numpy as np
import csv
import os
from state import State
from numpy.core.fromnumeric import cumsum

class QLearning():
    tag = 'X'
    values = dict()
    alpha = 0.5
    prev_stateStr = ""
    discount_factor = 0.9
    exp_factor = 0.8
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
            tempState = State(state)
            tempState.act_move(move)
            moves_op = tempState.get_valid_moves

            for move_op in moves_op:
                opState = State(tempState)
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

            temp = prev + QLearning.alpha*(R + curr - prev)
            
            QLearning.values[QLearning.prev_stateStr] = prev + QLearning.alpha*(R + curr - prev)

            if temp not in {1,-1,0,0.5,-0.5}:
                print(QLearning.values[QLearning.prev_stateStr])

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
        s = 'values1' + QLearning.tag + '.csv'
        try:
            os.remove(s)
        except:
            pass
        a = csv.writer(open(s, 'a', newline=''))

        for v, k in QLearning.values.items():
            a.writerow([v, k])

QLearning.load_values()

def select_move(cur_state, remain_time, winner = None):
    #print(QLearning.state2str(cur_state))
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        return QLearning.make_move_and_learn(cur_state, winner)
    return None
