import numpy as np
import csv
import os

from numpy.core import fromnumeric
from state import State, UltimateTTT_Move
from numpy.core.fromnumeric import cumsum

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

QLearning.load_values()

class MoveFirst:
    phase = 1
    target = -1
    oppostie = -1
    
    @staticmethod
    def findTarget(state: State):
        MoveFirst.target = [np.all(block == 0) for block in state.blocks].index(True)
        MoveFirst.oppostie = list(range(9))[-MoveFirst.target-1]
        print(MoveFirst.target)
        print(MoveFirst.oppostie)

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
        

def select_move(cur_state: State, remain_time, winner = None):
    valid_moves = cur_state.get_valid_moves 
    if len(valid_moves) != 0:
        if cur_state.player_to_move == State.X:
            return MoveFirst.selectMove(cur_state)
        else:
            return QLearning.make_move_and_learn(cur_state, winner)
    return None
import numpy as np
import state


def evaluate(move):
    return 0


def alphabeta(cur_state, depth, alpha, beta, player):
    if depth == 0:
        return evaluate(cur_state), None

    valid_moves = cur_state.get_valid_moves()
    if valid_moves == []:
        return evaluate(cur_state), None
    elif len(valid_moves) == 1:
        return evaluate(cur_state), valid_moves[0]
    else:
        if player == 1:
            value = float("-inf")
            for move in valid_moves:
                # cur_state chuyen trang thai??
                value = max(value, alphabeta(
                    move, depth - 1, alpha, beta, -player)[0])
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, move
        else:
            value = float("-inf")
            for move in valid_moves:
                value = min(value, alphabeta(
                    move, depth - 1, alpha, beta, -player)[0])
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, move


def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves

    if len(valid_moves) != 0:
        return np.random.choice(valid_moves)
    return None

    # value, move = alphabeta(cur_state, 4, float("-inf"), float("inf"), 1)
    # return move
