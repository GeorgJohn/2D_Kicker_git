import math

from kicker.CONST_BALL import Coordinate
from kicker.CONST_KICKER import COURT_HEIGHT
from kicker.CONST_GAME_FIGURES import FIGURE_FOOT_HEIGHT


class HumanStrategy:

    def __init__(self, kicker):
        self.kicker = kicker

    def next_move(self):
        if - math.pi / 2 < self.kicker.ball.angle < math.pi / 2:
            if self.kicker.ball.pos[Coordinate.Y] < COURT_HEIGHT / 2:
                new_pos_keeper = self.kicker.ball.pos[Coordinate.Y] - self.kicker.human_keeper.POSITION_ON_BAR - \
                                 FIGURE_FOOT_HEIGHT / 2
                new_pos_defender = self.kicker.ball.pos[Coordinate.Y] - \
                                   self.kicker.human_defender.POSITION_ON_BAR_DEFENDER_LEFT + FIGURE_FOOT_HEIGHT / 2

            elif self.kicker.ball.pos[Coordinate.Y] > COURT_HEIGHT / 2:
                new_pos_keeper = self.kicker.ball.pos[Coordinate.Y] - self.kicker.human_keeper.POSITION_ON_BAR \
                                 + FIGURE_FOOT_HEIGHT / 2
                new_pos_defender = self.kicker.ball.pos[Coordinate.Y] - \
                                   self.kicker.human_defender.POSITION_ON_BAR_DEFENDER_RIGHT - FIGURE_FOOT_HEIGHT / 2
            else:
                new_pos_keeper = self.kicker.ball.pos[Coordinate.Y] - self.kicker.human_keeper.POSITION_ON_BAR
                new_pos_defender = self.kicker.ball.pos[Coordinate.Y] - \
                                   self.kicker.human_defender.POSITION_ON_BAR_DEFENDER_RIGHT - FIGURE_FOOT_HEIGHT / 2
            if new_pos_keeper > self.kicker.human_keeper.MAX_POS_KEEPER:
                new_pos_keeper = self.kicker.human_keeper.MAX_POS_KEEPER
            elif new_pos_keeper < 0:
                new_pos_keeper = 0
            if new_pos_defender > self.kicker.human_defender.MAX_POS_DEFENDER:
                new_pos_defender = self.kicker.human_defender.MAX_POS_DEFENDER
            elif new_pos_defender < 0:
                new_pos_defender = 0
        else:
            new_pos_keeper = self.kicker.human_keeper.MAX_POS_KEEPER / 2
            new_pos_defender = self.kicker.human_defender.MAX_POS_DEFENDER / 2

        self.kicker.human_keeper.next_position = new_pos_keeper
        self.kicker.human_defender.next_position = new_pos_defender
        self.kicker.human_keeper.move_bar()
        self.kicker.human_defender.move_bar()

