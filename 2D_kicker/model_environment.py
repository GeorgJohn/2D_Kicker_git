import math
from kicker_simulation.CONST_KICKER import *
from kicker_simulation.CONST_BALL import *
from kicker_simulation.CONST_GAME_FIGURES import *


class Observation:

    def __init__(self):
        self._state = []

    def update_std(self, kicker):
        standardize_x_pos = kicker.ball.pos[Coordinate.X] / COURT_WIDTH
        standardize_y_pos = kicker.ball.pos[Coordinate.Y] / COURT_HEIGHT

        standardize_speed = kicker.ball.speed / BALL_MAX_SPEED
        standardize_angle = kicker.ball.angle / math.pi

        standardize_computer_keeper_pos = kicker.computer_keeper.position / MAX_POS_KEEPER
        standardize_computer_defender_pos = kicker.computer_defender.position / MAX_POS_DEFENDER
        standardize_human_gamer_pos = kicker.human_keeper.position / MAX_POS_KEEPER

        self._state = [kicker.get_score(), standardize_x_pos, standardize_y_pos, standardize_speed,
                       standardize_angle, standardize_computer_keeper_pos, standardize_computer_defender_pos,
                       standardize_human_gamer_pos]

    def update(self, kicker):
        x_pos = kicker.ball.pos[Coordinate.X]
        y_pos = kicker.ball.pos[Coordinate.Y]

        speed = kicker.ball.speed
        angle = kicker.ball.angle

        computer_keeper_pos = kicker.computer_keeper.position
        computer_defender_pos = kicker.computer_defender.position
        human_gamer_pos = kicker.human_keeper.position

        self._state = [kicker.get_score(), x_pos, y_pos, speed, angle,
                       computer_keeper_pos, computer_defender_pos, human_gamer_pos]

        # new_x_pos = ball.get_x_position() + math.cos(ball.get_angle()) * ball.get_speed()
        # new_y_pos = ball.get_y_position() + math.sin(ball.get_angle()) * ball.get_speed()
        # self._state = [kicker.get_score(), ball.get_x_position(), ball.get_y_position(), new_x_pos, new_y_pos,
        #                computer_gamer.get_position()]

    def get_state(self):
        return self._state


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
            score = self.get_state()[0][:]
            human_diff = score[0] - self.__old_score[0]
            computer_diff = score[1] - self.__old_score[1]
            if human_diff != 0:
                self.__reward = -4
            elif computer_diff != 0:
                self.__human_goal_counter += 1
                self.__reward = 4

            self.set_old_score(score)
        else:
            self.__reward = -0.1

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
        self.__old_score = score

    def get_goal_counter(self):
        return self.__human_goal_counter

    def set_goal_counter(self, count):
        self.__human_goal_counter = count

