import math

from kicker_simulation.model_game_bar import GameBar
from kicker_simulation.CONST_GAME_FIGURES import *
from kicker_simulation.CONST_BALL import Coordinate
from kicker_simulation.CONST_SIMULATION import SIMULATION_TIME_STEP
from kicker_simulation.CONST_SIMULATION import SHOOT_SPEED
from kicker_simulation.CONST_SIMULATION import BAR_SPEED
from kicker_simulation.CONST_KICKER import COURT_HEIGHT


class ComputerDefender(GameBar):
    """Konstanten"""
    NUMBER_OF_FIGURES = 2
    DISTANCE_FIGURES = DEFENDER_DISTANCE_BETWEEN_FIGURES
    POSITION_ON_BAR_DEFENDER_RIGHT = (COURT_HEIGHT - MAX_POS_DEFENDER - DISTANCE_FIGURES) / 2
    POSITION_ON_BAR_DEFENDER_LEFT = (COURT_HEIGHT - MAX_POS_DEFENDER + DISTANCE_FIGURES) / 2
    ABS_X_POSITION = BAR_POSITION_DEFENDER
    X_REFLECTION_PLANE = ABS_X_POSITION + FIGURE_FOOT_WIDTH + BALL_RADIUS
    X_OFFSET_REFLECTION_PLANE = FIGURE_FOOT_WIDTH + BALL_RADIUS
    Y_OFFSET_REFLECTION_PLANE = FIGURE_FOOT_HEIGHT / 2 + BALL_RADIUS

    def __init__(self):
        super().__init__(MAX_POS_DEFENDER / 2)

    def move_bar(self):
        self.next_position = round(self.next_position)
        self.position = round(self.position)
        if self.next_position != self.position:
            if 0 <= self.next_position <= MAX_POS_DEFENDER:
                if self.next_position > self.position:
                    new_temp_pos = self.position + BAR_SPEED * SIMULATION_TIME_STEP
                    if new_temp_pos > self.next_position:
                        self.position = self.next_position
                        self.next_position = -1
                    else:
                        self.position = new_temp_pos
                elif self.next_position < self.position:
                    new_temp_pos = self.position - BAR_SPEED * SIMULATION_TIME_STEP
                    if new_temp_pos < self.next_position:
                        self.position = self.next_position
                        self.next_position = -1
                    else:
                        self.position = new_temp_pos
            else:
                self.next_position = -1
        else:
            self.next_position = -1

    def reset_bar(self):
        self.next_position = -1
        self.position = MAX_POS_DEFENDER / 2

    def check_for_interaction(self, kicker):
        if kicker.ball.pos[Coordinate.X] > self.X_REFLECTION_PLANE:
            if kicker.ball.pos[Coordinate.X] - self.X_REFLECTION_PLANE <= (kicker.ball.pos[Coordinate.X] -
                                                                           kicker.ball.new_pos[Coordinate.X]):
                shoot = self.check_for_shoot(kicker)
            else:
                shoot = False
            side_collision = False
        elif self.X_REFLECTION_PLANE - kicker.ball.pos[Coordinate.X] < (kicker.ball.speed * SIMULATION_TIME_STEP +
                                                                        self.X_OFFSET_REFLECTION_PLANE):
            side_collision = self.check_for_side_collision(kicker)
            shoot = False
        else:
            side_collision = False
            shoot = False
        return shoot, side_collision

    def check_for_shoot(self, kicker):
        intersection = kicker.ball.pos[Coordinate.Y] - math.tan(kicker.ball.angle) * \
                       (kicker.ball.pos[Coordinate.X] - self.X_REFLECTION_PLANE)
        if self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT - self.Y_OFFSET_REFLECTION_PLANE < intersection \
                < self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT + self.Y_OFFSET_REFLECTION_PLANE:
            self.shoot(kicker, intersection, TeamHuman.DEFENDER_RIGHT)
            shoot = True
        elif self.position + self.POSITION_ON_BAR_DEFENDER_LEFT - self.Y_OFFSET_REFLECTION_PLANE < intersection \
                < self.position + self.POSITION_ON_BAR_DEFENDER_LEFT + self.Y_OFFSET_REFLECTION_PLANE:
            self.shoot(kicker, intersection, TeamHuman.DEFENDER_LEFT)
            shoot = True
        else:
            shoot = False
        return shoot

    def check_for_side_collision(self, kicker):
        if kicker.ball.angle > 0:
            if kicker.ball.pos[Coordinate.Y] < (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT -
                                                self.Y_OFFSET_REFLECTION_PLANE) < kicker.ball.new_pos[Coordinate.Y]:
                intersection = kicker.ball.pos[Coordinate.X] + (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT -
                                                                self.Y_OFFSET_REFLECTION_PLANE -
                                                                kicker.ball.pos[Coordinate.Y]) / \
                               math.tan(kicker.ball.angle)
                if self.ABS_X_POSITION < intersection < self.ABS_X_POSITION + self.X_OFFSET_REFLECTION_PLANE:
                    delta_t_collision = (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT -
                                         self.Y_OFFSET_REFLECTION_PLANE - kicker.ball.pos[Coordinate.Y]) / \
                                        (math.sin(kicker.ball.angle) * kicker.ball.speed)
                    kicker.ball.new_angle = - kicker.ball.angle
                    kicker.ball.new_pos[Coordinate.X] = (kicker.ball.pos[Coordinate.X] + math.cos(kicker.ball.angle) *
                                                         kicker.ball.speed * SIMULATION_TIME_STEP)
                    kicker.ball.new_pos[Coordinate.Y] = (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT -
                                                         self.Y_OFFSET_REFLECTION_PLANE +
                                                         math.sin(kicker.ball.new_angle) * kicker.ball.speed *
                                                         (SIMULATION_TIME_STEP - delta_t_collision))
                    side_collision = True
                else:
                    side_collision = False
            elif kicker.ball.pos[Coordinate.Y] < self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT \
                    - self.Y_OFFSET_REFLECTION_PLANE < kicker.ball.new_pos[Coordinate.Y]:
                intersection = kicker.ball.pos[Coordinate.X] + (self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT -
                                                                self.Y_OFFSET_REFLECTION_PLANE -
                                                                kicker.ball.pos[Coordinate.Y]) / \
                               math.tan(kicker.ball.angle)
                if self.ABS_X_POSITION < intersection < self.ABS_X_POSITION + self.X_OFFSET_REFLECTION_PLANE:
                    delta_t_collision = (self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT -
                                         self.Y_OFFSET_REFLECTION_PLANE - kicker.ball.pos[Coordinate.Y]) / \
                                        (math.sin(kicker.ball.angle) * kicker.ball.speed)
                    kicker.ball.new_angle = - kicker.ball.angle
                    kicker.ball.new_pos[Coordinate.X] = (kicker.ball.pos[Coordinate.X] + math.cos(kicker.ball.angle) *
                                                         kicker.ball.speed * SIMULATION_TIME_STEP)
                    kicker.ball.new_pos[Coordinate.Y] = (self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT -
                                                         self.Y_OFFSET_REFLECTION_PLANE +
                                                         math.sin(kicker.ball.new_angle) * kicker.ball.speed *
                                                         (SIMULATION_TIME_STEP - delta_t_collision))
                    side_collision = True
                else:
                    side_collision = False
            else:
                side_collision = False

        elif kicker.ball.angle < 0:
            if kicker.ball.pos[Coordinate.Y] > (self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT +
                                                self.Y_OFFSET_REFLECTION_PLANE) > kicker.ball.new_pos[Coordinate.Y]:
                intersection = kicker.ball.pos[Coordinate.X] + (kicker.ball.pos[Coordinate.Y] - self.position -
                                                                self.POSITION_ON_BAR_DEFENDER_RIGHT -
                                                                self.Y_OFFSET_REFLECTION_PLANE) / \
                               math.tan(kicker.ball.angle)
                if self.ABS_X_POSITION < intersection < self.ABS_X_POSITION + self.X_OFFSET_REFLECTION_PLANE:
                    delta_t_collision = (kicker.ball.pos[Coordinate.Y] - self.position -
                                         self.POSITION_ON_BAR_DEFENDER_RIGHT - self.Y_OFFSET_REFLECTION_PLANE) / \
                                        (math.sin(kicker.ball.angle) * kicker.ball.speed)
                    kicker.ball.new_angle = - kicker.ball.angle
                    kicker.ball.new_pos[Coordinate.X] = (kicker.ball.pos[Coordinate.X] + math.cos(kicker.ball.angle) *
                                                         kicker.ball.speed * SIMULATION_TIME_STEP)
                    kicker.ball.new_pos[Coordinate.Y] = (self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT +
                                                         self.Y_OFFSET_REFLECTION_PLANE +
                                                         math.sin(kicker.ball.new_angle) * kicker.ball.speed *
                                                         (SIMULATION_TIME_STEP - delta_t_collision))
                    side_collision = True
                else:
                    side_collision = False
            elif kicker.ball.pos[Coordinate.Y] > (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT +
                                                  self.Y_OFFSET_REFLECTION_PLANE) > kicker.ball.new_pos[Coordinate.Y]:
                intersection = kicker.ball.pos[Coordinate.X] + (kicker.ball.pos[Coordinate.Y] - self.position -
                                                                self.POSITION_ON_BAR_DEFENDER_LEFT -
                                                                self.Y_OFFSET_REFLECTION_PLANE) / \
                               math.tan(kicker.ball.angle)
                if self.ABS_X_POSITION < intersection < self.ABS_X_POSITION + self.X_OFFSET_REFLECTION_PLANE:
                    delta_t_collision = (kicker.ball.pos[Coordinate.Y] - self.position -
                                         self.POSITION_ON_BAR_DEFENDER_LEFT - self.Y_OFFSET_REFLECTION_PLANE) / \
                                        (math.sin(kicker.ball.angle) * kicker.ball.speed)
                    kicker.ball.new_angle = - kicker.ball.angle
                    kicker.ball.new_pos[Coordinate.X] = (kicker.ball.pos[Coordinate.X] + math.cos(kicker.ball.angle) *
                                                         kicker.ball.speed * SIMULATION_TIME_STEP)
                    kicker.ball.new_pos[Coordinate.Y] = (self.position + self.POSITION_ON_BAR_DEFENDER_LEFT +
                                                         self.Y_OFFSET_REFLECTION_PLANE +
                                                         math.sin(kicker.ball.new_angle) * kicker.ball.speed *
                                                         (SIMULATION_TIME_STEP - delta_t_collision))
                    side_collision = True
                else:
                    side_collision = False
            else:
                side_collision = False
        else:
            side_collision = False

        return side_collision

    def shoot(self, kicker, intersection, teammate):
        delta_t_collision = (kicker.ball.pos[Coordinate.X] - self.X_REFLECTION_PLANE) / \
                            (math.cos(kicker.ball.angle) * kicker.ball.speed)
        if teammate == TeamHuman.DEFENDER_RIGHT:
            shoot_offset = self.position + self.POSITION_ON_BAR_DEFENDER_RIGHT - intersection
        elif teammate == TeamHuman.DEFENDER_LEFT:
            shoot_offset = self.position + self.POSITION_ON_BAR_DEFENDER_LEFT - intersection
        else:
            raise ValueError('Uebergabeparameter (teammate) in model.HumanDefender.py falsch (Fkt. shoot())')
        kicker.ball.new_angle = ((math.pi / 3) * (- shoot_offset / (FIGURE_FOOT_HEIGHT / 2 + BALL_RADIUS)))
        kicker.ball.speed = SHOOT_SPEED
        kicker.ball.new_pos[Coordinate.Y] = (intersection + math.sin(kicker.ball.new_angle) * kicker.ball.speed *
                                             (SIMULATION_TIME_STEP - delta_t_collision))
        kicker.ball.new_pos[Coordinate.X] = (self.X_REFLECTION_PLANE + math.cos(kicker.ball.new_angle) *
                                             kicker.ball.speed * (SIMULATION_TIME_STEP - delta_t_collision))
