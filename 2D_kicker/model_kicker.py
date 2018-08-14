import math

from kicker_simulation.CONST_BALL import *
from kicker_simulation.CONST_KICKER import *
from kicker_simulation.CONST_SIMULATION import *
from kicker_simulation import model_ball
from kicker_simulation import model_human_keeper
from kicker_simulation import model_computer_keeper
from kicker_simulation import model_computer_defender


class Kicker:
    """Klassenvariablen"""
    borderline_x_min = BALL_RADIUS
    borderline_x_max = COURT_WIDTH - BALL_RADIUS
    borderline_y_min = BALL_RADIUS
    borderline_y_max = COURT_HEIGHT - BALL_RADIUS
    borderline_goal_bar_y_min = GOAL_BAR_POS + BALL_RADIUS
    borderline_goal_bar_y_max = GOAL_BAR_POS + GOAL_SIZE - BALL_RADIUS
    goal_line_human = COURT_WIDTH + BALL_RADIUS
    goal_line_computer = - BALL_RADIUS

    def __init__(self):
        self.ball = model_ball.Ball()
        self.ball.kick_off()
        self.human_keeper = model_human_keeper.HumanKeeper()
        self.computer_keeper = model_computer_keeper.ComputerKeeper()
        self.computer_defender = model_computer_defender.ComputerDefender()
        self.score = [0, 0]
        self.ball_in_goal_area = False
        self.terminal_state = False

    def update_model(self):
        self.terminal_state = False
        self.ball.move()
        self.human_keeper.check_for_interaction(self)
        self.computer_keeper.check_for_interaction(self)
        self.computer_defender.check_for_interaction(self)

        self.ball.update_angle()

        i = 0
        while i < 3 and not self.collision():
            i = i + 1

        if self.terminal_state or self.ball.speed == 0:
            self.ball.kick_off()
            self.terminal_state = True
        else:
            self.ball.update_all()

    def collision(self):
        new_pos_in_range = False
        if not self.ball_in_goal_area:
            if self.ball.new_pos[Coordinate.X] > self.borderline_x_max:
                if self.borderline_goal_bar_y_min < self.ball.new_pos[Coordinate.Y] < self.borderline_goal_bar_y_max:
                    x_pos_in_range = True
                    self.ball_in_goal_area = True
                else:
                    delta_t_collision = (self.borderline_x_max - self.ball.pos[Coordinate.X]) / \
                                        (math.cos(self.ball.angle) * self.ball.speed)
                    if self.ball.angle < 0:
                        self.ball.new_angle = - math.pi - self.ball.angle
                    elif self.ball.angle > 0:
                        self.ball.new_angle = math.pi - self.ball.angle
                    else:
                        self.ball.new_angle = math.pi
                    self.ball.new_pos[Coordinate.Y] = (self.ball.pos[Coordinate.Y] + math.sin(self.ball.angle) *
                                                       self.ball.speed * SIMULATION_TIME_STEP)
                    self.ball.new_pos[Coordinate.X] = (self.borderline_x_max + math.cos(self.ball.new_angle) *
                                                       self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
                    x_pos_in_range = False
            elif self.ball.new_pos[Coordinate.X] < self.borderline_x_min:
                if self.borderline_goal_bar_y_min < self.ball.new_pos[Coordinate.Y] < self.borderline_goal_bar_y_max:
                    x_pos_in_range = True
                    self.ball_in_goal_area = True
                else:
                    delta_t_collision = (self.borderline_x_min - self.ball.pos[Coordinate.X]) / \
                                        (math.cos(self.ball.angle) * self.ball.speed)
                    if self.ball.angle < 0:
                        self.ball.new_angle = - math.pi - self.ball.angle
                    elif self.ball.angle > 0:
                        self.ball.new_angle = math.pi - self.ball.angle
                    self.ball.new_pos[Coordinate.Y] = (self.ball.pos[Coordinate.Y] + math.sin(self.ball.angle) *
                                                       self.ball.speed * SIMULATION_TIME_STEP)
                    self.ball.new_pos[Coordinate.X] = (self.borderline_x_min + math.cos(self.ball.new_angle) *
                                                       self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
                    x_pos_in_range = False
            else:
                x_pos_in_range = True

            if self.ball.new_pos[Coordinate.Y] > self.borderline_y_max:
                delta_t_collision = (self.borderline_y_max - self.ball.pos[Coordinate.Y]) / \
                                    (math.sin(self.ball.angle) * self.ball.speed)
                self.ball.new_angle = - self.ball.angle
                self.ball.new_pos[Coordinate.X] = (self.ball.pos[Coordinate.X] + math.cos(self.ball.angle) *
                                                   self.ball.speed * SIMULATION_TIME_STEP)
                self.ball.new_pos[Coordinate.Y] = (self.borderline_y_max + math.sin(self.ball.new_angle) *
                                                   self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
                y_pos_in_range = False
            elif self.ball.new_pos[Coordinate.Y] < self.borderline_y_min:
                delta_t_collision = (self.ball.pos[Coordinate.Y] - self.borderline_y_min) / \
                                    (math.sin(self.ball.angle) * self.ball.speed)
                self.ball.new_angle = - self.ball.angle
                self.ball.new_pos[Coordinate.X] = (self.ball.pos[Coordinate.X] + math.cos(self.ball.angle) *
                                                   self.ball.speed * SIMULATION_TIME_STEP)
                self.ball.new_pos[Coordinate.Y] = (self.borderline_y_min + math.sin(self.ball.new_angle) *
                                                   self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
                y_pos_in_range = False
            else:
                y_pos_in_range = True
        else:
            if self.ball.new_pos[Coordinate.X] > self.goal_line_human:
                self.score_counter(Gamer.COMPUTER)
                self.terminal_state = True
                self.ball_in_goal_area = False
            elif self.ball.new_pos[Coordinate.X] < self.goal_line_computer:
                self.score_counter(Gamer.HUMAN)
                self.terminal_state = True
                self.ball_in_goal_area = False
            else:
                if self.ball.new_pos[Coordinate.Y] > self.borderline_goal_bar_y_max:
                    delta_t_collision = (self.borderline_goal_bar_y_max - self.ball.pos[Coordinate.Y]) \
                                        / (math.sin(self.ball.angle) * self.ball.speed)
                    self.ball.new_angle = - self.ball.angle
                    self.ball.new_pos[Coordinate.X] = (self.ball.pos[Coordinate.X] + math.cos(self.ball.angle) *
                                                       self.ball.speed * SIMULATION_TIME_STEP)
                    self.ball.new_pos[Coordinate.Y] = (self.borderline_goal_bar_y_max + math.sin(self.ball.new_angle) *
                                                       self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
                elif self.ball.new_pos[Coordinate.Y] < self.borderline_goal_bar_y_min:
                    delta_t_collision = (self.ball.pos[Coordinate.Y] - self.borderline_goal_bar_y_min) \
                                        / (math.sin(self.ball.angle) * self.ball.speed)
                    self.ball.new_angle = - self.ball.angle
                    self.ball.new_pos[Coordinate.X] = (self.ball.pos[Coordinate.X] + math.cos(self.ball.angle) *
                                                       self.ball.speed * SIMULATION_TIME_STEP)
                    self.ball.new_pos[Coordinate.Y] = (self.borderline_goal_bar_y_min + math.sin(self.ball.new_angle) *
                                                       self.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
            y_pos_in_range = True
            x_pos_in_range = True

        if y_pos_in_range and x_pos_in_range:
            new_pos_in_range = True

        return new_pos_in_range

    def score_counter(self, gamer):
        if gamer == Gamer.HUMAN:
            self.score[Gamer.HUMAN] = self.score[Gamer.HUMAN] + 1
        elif gamer == Gamer.COMPUTER:
            self.score[Gamer.COMPUTER] = self.score[Gamer.COMPUTER] + 1

    def get_score(self):
        return self.score

    def reset_score_counter(self):
        self.score = [0, 0]
