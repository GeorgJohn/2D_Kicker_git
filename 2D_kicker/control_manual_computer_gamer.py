from kicker_simulation.CONST_SIMULATION import *


class ManualKeeperController:
    def __init__(self, kicker):
        self.kicker = kicker
        self.move_up_flag = False
        self.move_down_flag = False

    def move_bar(self):
        if self.move_up_flag:
            self.move_down()
        elif self.move_down_flag:
            self.move_up()
        else:
            self.reset_move_bar()

    # keydown handler
    def move_down(self):
        self.kicker.computer_keeper.next_position = \
            self.kicker.computer_keeper.position + (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_keeper.move_bar()

    def move_up(self):
        self.kicker.computer_keeper.next_position = \
            self.kicker.computer_keeper.position - (BAR_SPEED * SIMULATION_TIME_STEP)
        self.kicker.computer_keeper.move_bar()

    def reset_move_bar(self):
        self.move_up_flag = False
        self.move_down_flag = False
        self.kicker.computer_keeper.next_position = -1

    def set_move_up(self):
        self.move_up_flag = True
        self.move_down_flag = False

    def set_move_down(self):
        self.move_up_flag = False
        self.move_down_flag = True
