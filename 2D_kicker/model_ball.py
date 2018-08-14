import random
import math

from kicker_simulation.CONST_KICKER import COURT_WIDTH
from kicker_simulation.CONST_KICKER import COURT_HEIGHT
from kicker_simulation.CONST_GAME_FIGURES import BAR_POSITION_KEEPER
from kicker_simulation.CONST_GAME_FIGURES import BAR_POSITION_DEFENDER
from kicker_simulation.CONST_BALL import *
from kicker_simulation.CONST_SIMULATION import *


class Ball:
    def __init__(self, x_pos=0, y_pos=0, z_pos=0, speed=0.0, angle=0.0, omega_z=0.0):
        self.pos = [x_pos, y_pos, z_pos]
        self.new_pos = [x_pos, y_pos, z_pos]
        self.angle = angle
        self.new_angle = angle
        self.speed = speed
        self.omega = omega_z

    def move(self):
        self.new_pos[Coordinate.X] = self.pos[Coordinate.X] + math.cos(self.angle) * self.speed * SIMULATION_TIME_STEP
        self.new_pos[Coordinate.Y] = self.pos[Coordinate.Y] + math.sin(self.angle) * self.speed * SIMULATION_TIME_STEP

    def update_angle(self):
        self.angle = self.new_angle

    def update_all(self):
        self.speed = self.speed - 0.01 * 9810 * SIMULATION_TIME_STEP
        if self.speed <= 0:
            self.speed = 0

        self.pos[Coordinate.X] = self.new_pos[Coordinate.X]
        self.pos[Coordinate.Y] = self.new_pos[Coordinate.Y]
        self.angle = self.new_angle

    def kick_off(self):
        """Ball startet immer von der Mitte mit einem variablen Winkel"""
        # self.pos[Coordinate.X] = self.new_pos[Coordinate.X] = COURT_WIDTH / 2
        # self.pos[Coordinate.Y] = self.new_pos[Coordinate.Y] = COURT_HEIGHT / 2
        #
        # if random.randint(0, 1):
        #     self.new_angle = random.uniform(- 0.1, 0.1)
        #     if self.new_angle > 0:
        #         self.angle = self.new_angle = self.new_angle - math.pi
        #     else:
        #         self.angle = self.new_angle = self.new_angle + math.pi
        # else:
        #     self.new_angle = random.uniform(- 0.1, 0.1)
        # self.speed = KICK_OFF_SPEED

        """Ball startet auf einem zufälligen Punkt der y-Achse in der Mitte des Spielfeldes
           der Startwinkel ist aber immer so das der Ball genau auf die mitte des Tores kommt"""
        self.pos[Coordinate.X] = self.new_pos[Coordinate.X] = COURT_WIDTH - BAR_POSITION_DEFENDER
        self.pos[Coordinate.Y] = self.new_pos[Coordinate.Y] = random.randint(BALL_RADIUS, COURT_HEIGHT - BALL_RADIUS)

        delta_y = COURT_HEIGHT / 2 - self.pos[Coordinate.Y]
        self.angle = self.new_angle = math.pi - math.atan2(delta_y, COURT_WIDTH - BAR_POSITION_DEFENDER)
        # if random.randint(0, 1):
        #     self.angle = self.new_angle
        # else:
        #     self.angle = self.new_angle = math.pi - self.new_angle

        self.speed = KICK_OFF_SPEED

    def get_gradient_goal_to_ball(self):
        delta_y = COURT_HEIGHT / 2 - self.pos[Coordinate.Y]
        delta_x = COURT_WIDTH - self.pos[Coordinate.X]
        return delta_y / delta_x

    def get_x_distance_human_keeper_to_ball(self):
        return COURT_WIDTH - BAR_POSITION_KEEPER - self.pos[Coordinate.X]

    def get_x_distance_human_defender_to_ball(self):
        return COURT_WIDTH - BAR_POSITION_DEFENDER - self.pos[Coordinate.X]
