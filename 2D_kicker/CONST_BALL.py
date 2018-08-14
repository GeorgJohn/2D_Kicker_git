from enum import IntEnum

"""Ball"""
BALL_RADIUS = 17
BALL_DIAMETER = 2 * BALL_RADIUS
BALL_MASS = 23.13
BALL_MAX_SPEED = 1000


class Coordinate(IntEnum):
    X = 0
    Y = 1
    Z = 2
