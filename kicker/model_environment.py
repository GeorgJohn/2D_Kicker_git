import math
import collections

from kicker.CONST_KICKER import *
from kicker.CONST_BALL import *
from kicker.CONST_GAME_FIGURES import *


class Observation:

    MAX_LEN_BUFFER = 4

    def __init__(self):
        self._score = [0, 0]
        self._state_buffer = collections.deque(maxlen=Observation.MAX_LEN_BUFFER)
        self._index_buffer = 0
        self._state = []

    def update_std(self, kicker):
        standardize_x_pos = kicker.ball.pos[Coordinate.X] / COURT_WIDTH
        standardize_y_pos = kicker.ball.pos[Coordinate.Y] / COURT_HEIGHT

        standardize_speed = kicker.ball.speed / BALL_MAX_SPEED
        standardize_angle = kicker.ball.angle / math.pi

        standardize_computer_keeper_pos = kicker.computer_keeper.position / MAX_POS_KEEPER
        standardize_human_gamer_pos = kicker.human_keeper.position / MAX_POS_KEEPER

        self._state = [standardize_x_pos, standardize_y_pos, standardize_speed, standardize_angle,
                       standardize_computer_keeper_pos, standardize_human_gamer_pos]

        self._state_buffer.append(self._state)

        self._index_buffer += 1
        if self._index_buffer >= Observation.MAX_LEN_BUFFER:
            self._index_buffer = 0

        self._score = kicker.get_score()

    def get_state(self):
        state = []
        tmp_index = self._index_buffer
        for i in range(Observation.MAX_LEN_BUFFER):
            tmp_index -= 1
            if tmp_index < 0:
                tmp_index = Observation.MAX_LEN_BUFFER - 1
            for j in self._state_buffer[tmp_index]:
                state.append(j)
        return state


class Environment(Observation):

    def __init__(self):
        super().__init__()
        self.__reward = 0
        self.__old_score = [0, 0]
        self.__human_goal_counter = 0
        self.__done = False
        self.__enable_view = False

    def calc_reward(self):
        # self.__reward += 0
        # if flag:
        #     self.__reward = 1
        if self.__done:
            human_diff = self._score[0] - self.__old_score[0]
            computer_diff = self._score[1] - self.__old_score[1]
            if human_diff != 0:
                self.__reward = -1
            elif computer_diff != 0:
                self.__human_goal_counter += 1
                self.__reward = 1

            self.set_old_score(self._score)
        else:
            self.__reward = 0.01

    def get_reward(self):
        return self.__reward

    def set_reward(self, reward):
        self.__reward = reward

    def get_done(self):
        return self.__done

    def set_done(self, boolean):
        self.__done = boolean

    def set_enable_view(self, boolean):
        self.__enable_view = boolean

    def get_enable_view(self):
        return self.__enable_view

    def get_observation(self):
        return self.get_state()

    def set_old_score(self, score):
        self.__old_score = score.copy()

    def get_goal_counter(self):
        return self.__human_goal_counter

    def set_goal_counter(self, count):
        self.__human_goal_counter = count

