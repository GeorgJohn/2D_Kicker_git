import math

from kicker_simulation.CONST_BALL import Coordinate


class HumanStrategy:

    def __init__(self, kicker):
        self.kicker = kicker

    def next_move(self):
        if - math.pi / 2 < self.kicker.ball.angle < math.pi / 2:
            new_pos = self.kicker.ball.pos[Coordinate.Y] - self.kicker.human_keeper.POSITION_ON_BAR
            if new_pos > self.kicker.human_keeper.MAX_POS_KEEPER:
                new_pos = self.kicker.human_keeper.MAX_POS_KEEPER
            elif new_pos < 0:
                new_pos = 0
        else:
            new_pos = self.kicker.human_keeper.MAX_POS_KEEPER / 2

        self.kicker.human_keeper.next_position = new_pos
        self.kicker.human_keeper.move_bar()
