from kicker.CONST_KICKER import *
from kicker.CONST_GAME_FIGURES import *


"""Fenster"""
SCREEN_WIDTH = 1360
SCREEN_HIGH = 840
MARGIN_LEFT = 80
MARGIN_RIGHT = 80
MARGIN_BUTTON = 30
MARGIN_TOP = 130

"""Infofenster"""
INFO_DISP_HIGH = 100

"""Grundfarben"""
WITHE = (255, 255, 255)
GRAY = (191, 191, 191)
BLACK = (0, 0, 0)
DARK_BLUE = (11, 36, 158)

"""Spielfeld"""
FIELD_COLOUR = (12, 128, 40)
COURT_LINE_THICKNESS = 8
COURT_LINE_POINT_RADIUS = 10
COURT_LINE_CIRCLE_RADIUS = 103

"""Offset zum Zeichnen der Tore"""
HUMAN_GOAL_OFFSET_X = MARGIN_LEFT - GOAL_WIDTH
HUMAN_GOAL_OFFSET_Y = MARGIN_TOP + GOAL_BAR_POS

COMPUTER_GOAL_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT
COMPUTER_GOAL_OFFSET_Y = HUMAN_GOAL_OFFSET_Y


"""Offset Positionen zum zeichnen der Stangen im Screen"""
"""Spielstangen des Menschen"""
HUMAN_BAR_KEEPER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_KEEPER - BAR_WIDTH / 2
HUMAN_BAR_KEEPER_OFFSET_Y = MARGIN_TOP

HUMAN_BAR_DEFENDER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_DEFENDER - BAR_WIDTH / 2
HUMAN_BAR_DEFENDER_OFFSET_Y = MARGIN_TOP

HUMAN_BAR_MIDFIELDER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_MIDFIELDER - BAR_WIDTH / 2
HUMAN_BAR_MIDFIELDER_OFFSET_Y = MARGIN_TOP

HUMAN_BAR_FORWARD_OFFSET_X = MARGIN_LEFT + BAR_POSITION_FORWARD - BAR_WIDTH / 2
HUMAN_BAR_FORWARD_OFFSET_Y = MARGIN_TOP

"""Spielstangen des Computers"""
COMPUTER_BAR_KEEPER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_KEEPER - BAR_WIDTH / 2
COMPUTER_BAR_KEEPER_OFFSET_Y = MARGIN_TOP

COMPUTER_BAR_DEFENDER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_DEFENDER - BAR_WIDTH / 2
COMPUTER_BAR_DEFENDER_OFFSET_Y = MARGIN_TOP

COMPUTER_BAR_MIDFIELDER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_MIDFIELDER - BAR_WIDTH / 2
COMPUTER_BAR_MIDFIELDER_OFFSET_Y = MARGIN_TOP

COMPUTER_BAR_FORWARD_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_FORWARD - BAR_WIDTH / 2
COMPUTER_BAR_FORWARD_OFFSET_Y = MARGIN_TOP


"""Offset Positionen zum zeichnen der Figuren im Screen"""
HUMAN_KEEPER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_KEEPER
HUMAN_KEEPER_OFFSET_Y = (COURT_HEIGHT - MAX_POS_KEEPER) / 2 + MARGIN_TOP

HUMAN_DEFENDER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_DEFENDER
HUMAN_DEFENDER_OFFSET_Y = (COURT_HEIGHT - MAX_POS_DEFENDER - DEFENDER_DISTANCE_BETWEEN_FIGURES) / 2 + MARGIN_TOP

HUMAN_MIDFIELDER_OFFSET_X = MARGIN_LEFT + BAR_POSITION_MIDFIELDER
HUMAN_MIDFIELDER_OFFSET_Y = HUMAN_DEFENDER_OFFSET_Y

HUMAN_FORWARD_OFFSET_X = MARGIN_LEFT + BAR_POSITION_FORWARD
HUMAN_FORWARD_OFFSET_Y = HUMAN_DEFENDER_OFFSET_Y

COMPUTER_KEEPER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_KEEPER
COMPUTER_KEEPER_OFFSET_Y = HUMAN_KEEPER_OFFSET_Y

COMPUTER_DEFENDER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_DEFENDER
COMPUTER_DEFENDER_OFFSET_Y = HUMAN_DEFENDER_OFFSET_Y

COMPUTER_MIDFIELDER_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_MIDFIELDER
COMPUTER_MIDFIELDER_OFFSET_Y = HUMAN_DEFENDER_OFFSET_Y

COMPUTER_FORWARD_OFFSET_X = SCREEN_WIDTH - MARGIN_RIGHT - BAR_POSITION_FORWARD
COMPUTER_FORWARD_OFFSET_Y = HUMAN_DEFENDER_OFFSET_Y


class GameBorder(IntEnum):
    TOP = 1
    BUTTON = 2
    LEFT = 3
    RIGHT = 4