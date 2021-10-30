import numpy as np
from state import State
from copy import deepcopy
from keras import layers as Kl
from keras import models as Km
import os
from pathlib import Path

class Dlearning():
    tag = State.O
    value_model = None
    alpha = 0.7
    exp_factor = 1

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