import pygame
import random

from kicker_simulation.CONST_SIMULATION import *
from kicker_simulation.CONST_KICKER import *

from kicker_simulation.model_environment import Environment
from kicker_simulation.view_game import View
from kicker_simulation.model_kicker import Kicker
from kicker_simulation.control_human_automatic_strategy import HumanStrategy

# KEEPER_START_POS = MAX_POS_KEEPER / 2
BALL_START_POS_X = COURT_WIDTH / 2
BALL_START_POS_Y = COURT_HEIGHT / 2
BALL_START_SPEED = 1.0 * 500
BAR_SPEED = 1.0 * 500
TIME_STEP = 1 / 60


class Action(IntEnum):
    UP_KEEPER = 0
    DOWN_KEEPER = 1

    UP_DEFENDER = 2
    DOWN_DEFENDER = 3

    NOOP_DEFENDER_KEEPER = 4
    UP_DEFENDER_KEEPER = 5
    DOWN_DEFENDER_KEEPER = 6


class ActionHandler:

    def __init__(self, kicker):
        self.kicker = kicker

    def move_bar(self, action):
        if action == Action.UP_KEEPER:
            self.move_up_keeper()
            self.no_move_defender()
        elif action == Action.DOWN_KEEPER:
            self.move_down_keeper()
            self.no_move_defender()
        elif action == Action.UP_DEFENDER:
            self.move_up_defender()
            self.no_move_keeper()
        elif action == Action.DOWN_DEFENDER:
            self.move_down_defender()
            self.no_move_keeper()
        elif action == Action.NOOP_DEFENDER_KEEPER:
            self.no_move_keeper()
            self.no_move_defender()
        elif action == Action.UP_DEFENDER_KEEPER:
            self.move_up_keeper()
            self.move_up_defender()
        elif action == Action.DOWN_DEFENDER_KEEPER:
            self.move_down_keeper()
            self.move_down_defender()
        else:
            print("undefined action !!!")

    def move_up_keeper(self):
        self.kicker.computer_keeper.next_position = \
            self.kicker.computer_keeper.position - (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_keeper.move_bar()

    def move_down_keeper(self):
        self.kicker.computer_keeper.next_position = \
            self.kicker.computer_keeper.position + (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_keeper.move_bar()

    def no_move_keeper(self):
        self.kicker.computer_keeper.next_position = -1

    def move_up_defender(self):
        self.kicker.computer_defender.next_position = \
            self.kicker.computer_defender.position - (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_defender.move_bar()

    def move_down_defender(self):
        self.kicker.computer_defender.next_position = \
            self.kicker.computer_defender.position + (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_defender.move_bar()

    def no_move_defender(self):
        self.kicker.computer_defender.next_position = -1


class EnvironmentController:

    def __init__(self):
        self.env = Environment()
        self.kicker = Kicker()
        self.human_strategy = HumanStrategy(self.kicker)
        self.action_handler = ActionHandler(self.kicker)
        self.create_view = False
        self.view = None

    def reset(self):
        self.env.set_done(False)
        self.kicker.terminal_state = True
        if self.kicker.get_score()[0] >= 10 or self.kicker.get_score()[1] >= 10:
            self.kicker.reset_score_counter()
            self.env.set_old_score([0, 0])

        self.kicker.computer_keeper.reset_bar()
        self.kicker.computer_defender.reset_bar()
        # self.kicker.computer_keeper.position = random.randint(0, MAX_POS_KEEPER)
        # self.kicker.computer_defender.position = random.randint(0, MAX_POS_DEFENDER)
        self.kicker.human_keeper.reset_bar()
        self.env.update_std(self.kicker)
        self.env.set_reward(0)
        return [self.env.get_observation(), self.env.get_reward(), self.env.get_done()]

    def render(self):
        if not self.create_view:
            self.view = View()
            self.create_view = True
        self.view.display_all(self.kicker)

        pygame.display.flip()

    def step(self, action):

        for i in range(2):
            self.human_strategy.next_move()
            self.action_handler.move_bar(action)
            self.kicker.update_model()
            self.env.update_std(self.kicker)
            if self.check_for_done():
                break

        self.env.calc_reward()
        return [self.env.get_observation(), self.env.get_reward(), self.env.get_done()]

    def check_for_done(self):
        if self.kicker.terminal_state:
            self.env.set_done(True)
        else:
            self.env.set_done(False)

        return self.env.get_done()

    @staticmethod
    def get_random_action():
        return random.randint(0, 8)

    def get_goal_counter(self):
        return self.env.get_goal_counter()

    def set_goal_counter(self, count):
        self.env.set_goal_counter(count)
