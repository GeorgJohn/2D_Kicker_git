import pygame

from kicker.view_game import View
from kicker.model_kicker import Kicker
from kicker.control_human_automatic_strategy import HumanStrategy
from kicker.control_manual_computer_gamer import ManualKeeperController

clock = pygame.time.Clock()

view = View()
kicker = Kicker()
human_strategy = HumanStrategy(kicker)
manual_com_keeper = ManualKeeperController(kicker)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                manual_com_keeper.set_move_up()
            elif event.key == pygame.K_UP:
                manual_com_keeper.set_move_down()
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_UP, pygame.K_DOWN):
                manual_com_keeper.reset_move_bar()
        if event.type == pygame.QUIT:
            running = False

    human_strategy.next_move()
    manual_com_keeper.move_bar()
    kicker.update_model()

    view.display_all(kicker)

    clock.tick_busy_loop(60)

    pygame.display.flip()
