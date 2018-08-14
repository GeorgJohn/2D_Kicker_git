import pygame
from kicker_simulation.CONST_VIEW import *
from kicker_simulation.CONST_BALL import Coordinate


class View:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HIGH))
        self.smallfont = pygame.font.SysFont("comicsansms", 72)
        pygame.display.set_caption('ProCK Simulation')

    def display_empty_screen(self):
        self.screen.fill(FIELD_COLOUR)

    def display_ball(self, ball):
        pygame.draw.circle(self.screen, WITHE,
                           (((-1) * round(ball.pos[Coordinate.X]) + MARGIN_LEFT + COURT_WIDTH),
                            (round(ball.pos[Coordinate.Y]) + MARGIN_TOP)), BALL_RADIUS)

    def display_gamer_bars(self, keeper=None, defender=None, midfielder=None, forward=None):
        if keeper:
            pygame.draw.rect(self.screen, GRAY, [HUMAN_BAR_KEEPER_OFFSET_X, HUMAN_BAR_KEEPER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])
            pygame.draw.rect(self.screen, GRAY, [COMPUTER_BAR_KEEPER_OFFSET_X, COMPUTER_BAR_KEEPER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])

        if defender:
            pygame.draw.rect(self.screen, GRAY, [HUMAN_BAR_DEFENDER_OFFSET_X, HUMAN_BAR_DEFENDER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])
            pygame.draw.rect(self.screen, GRAY, [COMPUTER_BAR_DEFENDER_OFFSET_X, COMPUTER_BAR_DEFENDER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])

        if midfielder:
            pygame.draw.rect(self.screen, GRAY, [HUMAN_BAR_MIDFIELDER_OFFSET_X, HUMAN_BAR_MIDFIELDER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])
            pygame.draw.rect(self.screen, GRAY, [COMPUTER_BAR_MIDFIELDER_OFFSET_X, COMPUTER_BAR_MIDFIELDER_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])

        if forward:
            pygame.draw.rect(self.screen, GRAY, [HUMAN_BAR_FORWARD_OFFSET_X, HUMAN_BAR_FORWARD_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])
            pygame.draw.rect(self.screen, GRAY, [COMPUTER_BAR_FORWARD_OFFSET_X, COMPUTER_BAR_FORWARD_OFFSET_Y,
                                                 BAR_WIDTH, COURT_HEIGHT])

    def display_human_figures(self, hum_keeper=None, hum_defender=None, hum_midfielder=None, hum_forward=None):
        if hum_keeper:
            self.draw_human_keeper(HUMAN_KEEPER_OFFSET_X,
                                   round(hum_keeper.position) + HUMAN_KEEPER_OFFSET_Y)
        if hum_defender:
            self.draw_human_defender(HUMAN_DEFENDER_OFFSET_X,
                                     round(hum_defender.position) + HUMAN_DEFENDER_OFFSET_Y)
        if hum_midfielder:
            self.draw_human_midfielder(HUMAN_MIDFIELDER_OFFSET_X,
                                       round(hum_midfielder.position) + HUMAN_MIDFIELDER_OFFSET_Y)
        if hum_forward:
            self.draw_human_forward(HUMAN_FORWARD_OFFSET_X,
                                    round(hum_forward.position) + HUMAN_FORWARD_OFFSET_Y)

    def display_computer_figures(self, com_keeper=None, com_defender=None, com_midfielder=None, com_forward=None):
        if com_keeper:
            self.draw_computer_keeper(COMPUTER_KEEPER_OFFSET_X,
                                      round(com_keeper.position) + COMPUTER_KEEPER_OFFSET_Y)
        if com_defender:
            self.draw_computer_defender(COMPUTER_DEFENDER_OFFSET_X,
                                        round(com_defender.position) + COMPUTER_DEFENDER_OFFSET_Y)
        if com_midfielder:
            self.draw_computer_midfielder(COMPUTER_MIDFIELDER_OFFSET_X,
                                          round(com_midfielder.position) + COMPUTER_MIDFIELDER_OFFSET_Y)
        if com_forward:
            self.draw_computer_forward(COMPUTER_FORWARD_OFFSET_X,
                                       round(com_forward.position) + COMPUTER_FORWARD_OFFSET_Y)

    def draw_computer_figure(self, x_pos, y_pos):
        pygame.draw.rect(self.screen, BLACK, [x_pos - FIGURE_FOOT_WIDTH, y_pos-FIGURE_FOOT_HEIGHT / 2,
                                              FIGURE_FOOT_WIDTH, FIGURE_FOOT_HEIGHT])
        pygame.draw.rect(self.screen, BLACK, [x_pos - FIGURE_BODY_WIDTH / 2, y_pos - FIGURE_BODY_HEIGHT / 2,
                                              FIGURE_BODY_WIDTH, FIGURE_BODY_HEIGHT])
        pygame.draw.rect(self.screen, BLACK, [x_pos, y_pos - FIGURE_HEAD_HEIGHT / 2,
                                              FIGURE_HEAD_WIDTH, FIGURE_HEAD_HEIGHT])

    def draw_human_figure(self, x_pos, y_pos):
        pygame.draw.rect(self.screen, DARK_BLUE, [x_pos, y_pos-FIGURE_FOOT_HEIGHT / 2,
                                                  FIGURE_FOOT_WIDTH, FIGURE_FOOT_HEIGHT])
        pygame.draw.rect(self.screen, DARK_BLUE, [x_pos - FIGURE_BODY_WIDTH / 2, y_pos - FIGURE_BODY_HEIGHT / 2,
                                                  FIGURE_BODY_WIDTH, FIGURE_BODY_HEIGHT])
        pygame.draw.rect(self.screen, DARK_BLUE, [x_pos - FIGURE_HEAD_WIDTH, y_pos - FIGURE_HEAD_HEIGHT / 2,
                                                  FIGURE_HEAD_WIDTH, FIGURE_HEAD_HEIGHT])

    def draw_computer_keeper(self, x_pos, y_pos):
        self.draw_computer_figure(x_pos, y_pos)

    def draw_computer_defender(self, x_pos, y_pos):
        self.draw_computer_figure(x_pos, y_pos)
        self.draw_computer_figure(x_pos, y_pos + DEFENDER_DISTANCE_BETWEEN_FIGURES)

    def draw_computer_midfielder(self, x_pos, y_pos):
        self.draw_computer_figure(x_pos, y_pos)
        self.draw_computer_figure(x_pos, y_pos + MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_computer_figure(x_pos, y_pos + 2 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_computer_figure(x_pos, y_pos + 3 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_computer_figure(x_pos, y_pos + 4 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)

    def draw_computer_forward(self, x_pos, y_pos):
        self.draw_computer_figure(x_pos, y_pos)
        self.draw_computer_figure(x_pos, y_pos + FORWARD_DISTANCE_BETWEEN_FIGURES)
        self.draw_computer_figure(x_pos, y_pos + 2 * FORWARD_DISTANCE_BETWEEN_FIGURES)

    def draw_human_keeper(self, x_pos, y_pos):
        self.draw_human_figure(x_pos, y_pos)

    def draw_human_defender(self, x_pos, y_pos):
        self.draw_human_figure(x_pos, y_pos)
        self.draw_human_figure(x_pos, y_pos + DEFENDER_DISTANCE_BETWEEN_FIGURES)

    def draw_human_midfielder(self, x_pos, y_pos):
        self.draw_human_figure(x_pos, y_pos)
        self.draw_human_figure(x_pos, y_pos + MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_human_figure(x_pos, y_pos + 2 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_human_figure(x_pos, y_pos + 3 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)
        self.draw_human_figure(x_pos, y_pos + 4 * MIDFIELDER_DISTANCE_BETWEEN_FIGURES)

    def draw_human_forward(self, x_pos, y_pos):
        self.draw_human_figure(x_pos, y_pos)
        self.draw_human_figure(x_pos, y_pos + FORWARD_DISTANCE_BETWEEN_FIGURES)
        self.draw_human_figure(x_pos, y_pos + 2 * FORWARD_DISTANCE_BETWEEN_FIGURES)

    def display_goal(self):
        pygame.draw.rect(self.screen, BLACK, [HUMAN_GOAL_OFFSET_X, HUMAN_GOAL_OFFSET_Y, GOAL_WIDTH, GOAL_SIZE])
        pygame.draw.rect(self.screen, BLACK, [COMPUTER_GOAL_OFFSET_X, COMPUTER_GOAL_OFFSET_Y, GOAL_WIDTH, GOAL_SIZE])

    def display_court_line(self):
        pygame.draw.circle(self.screen, WITHE, (600 + MARGIN_LEFT, 340 + MARGIN_TOP), COURT_LINE_POINT_RADIUS)
        pygame.draw.circle(self.screen, WITHE, (830 + MARGIN_LEFT, 340 + MARGIN_TOP), COURT_LINE_POINT_RADIUS)
        pygame.draw.circle(self.screen, WITHE, (370 + MARGIN_LEFT, 340 + MARGIN_TOP), COURT_LINE_POINT_RADIUS)
        pygame.draw.line(self.screen, WITHE, (600 + MARGIN_LEFT, MARGIN_TOP),
                         (600 + MARGIN_LEFT, 240 + MARGIN_TOP), COURT_LINE_THICKNESS)
        pygame.draw.line(self.screen, WITHE, (600 + MARGIN_LEFT, 440 + MARGIN_TOP),
                         (600 + MARGIN_LEFT, SCREEN_HIGH - MARGIN_BUTTON), COURT_LINE_THICKNESS)
        pygame.draw.rect(self.screen, WITHE, [-10 + MARGIN_LEFT, 140 + MARGIN_TOP, 290, 400], COURT_LINE_THICKNESS)
        pygame.draw.rect(self.screen, WITHE, [920 + MARGIN_LEFT, 140 + MARGIN_TOP, 290, 400], COURT_LINE_THICKNESS)
        pygame.draw.rect(self.screen, WITHE, [-10 + MARGIN_LEFT, 205 + MARGIN_TOP, 160, 270], COURT_LINE_THICKNESS)
        pygame.draw.rect(self.screen, WITHE, [1050 + MARGIN_LEFT, 205 + MARGIN_TOP, 160, 270], COURT_LINE_THICKNESS)
        pygame.draw.arc(self.screen, WITHE, [135 + MARGIN_LEFT, 245 + MARGIN_TOP, 200, 200],
                        5.2, 1.1, COURT_LINE_THICKNESS)
        pygame.draw.arc(self.screen, WITHE, [865 + MARGIN_LEFT, 240 + MARGIN_TOP, 200, 200],
                        2.05, 4.25, COURT_LINE_THICKNESS)
        pygame.draw.circle(self.screen, WITHE, (600 + MARGIN_LEFT, 345 + MARGIN_TOP), COURT_LINE_CIRCLE_RADIUS,
                           COURT_LINE_THICKNESS)

    def display_info(self):
        pygame.draw.rect(self.screen, WITHE, [0, 0, SCREEN_WIDTH, MARGIN_TOP])
        pygame.draw.rect(self.screen, WITHE, [0, SCREEN_HIGH - MARGIN_BUTTON, SCREEN_WIDTH, MARGIN_BUTTON])
        pygame.draw.rect(self.screen, WITHE, [0, MARGIN_TOP, MARGIN_LEFT, COURT_HEIGHT])
        pygame.draw.rect(self.screen, WITHE, [SCREEN_WIDTH - MARGIN_RIGHT, MARGIN_TOP, MARGIN_RIGHT, COURT_HEIGHT])
        pygame.draw.line(self.screen, BLACK, (MARGIN_LEFT - 2, MARGIN_TOP - 2),
                         (COURT_WIDTH + MARGIN_LEFT, MARGIN_TOP - 2), 2)
        pygame.draw.line(self.screen, BLACK, (MARGIN_LEFT - 2, SCREEN_HIGH - MARGIN_BUTTON),
                         (COURT_WIDTH + MARGIN_LEFT, SCREEN_HIGH - MARGIN_BUTTON), 2)
        pygame.draw.line(self.screen, BLACK, (MARGIN_LEFT - 2, MARGIN_TOP - 2),
                         (MARGIN_LEFT - 2, SCREEN_HIGH - MARGIN_BUTTON), 2)
        pygame.draw.line(self.screen, BLACK, (COURT_WIDTH + MARGIN_LEFT, MARGIN_TOP - 2),
                         (COURT_WIDTH + MARGIN_LEFT, SCREEN_HIGH - MARGIN_BUTTON), 2)

    def display_score(self, score):
        text = self.smallfont.render(str(score[Gamer.HUMAN]) + " : " + str(score[Gamer.COMPUTER]), True, BLACK)
        text_width = text.get_rect().width
        self.screen.blit(text, [SCREEN_WIDTH / 2 - text_width / 2, 0])

    def display_all(self, kicker):
        self.display_empty_screen()
        self.display_court_line()
        self.display_info()
        self.display_ball(kicker.ball)
        self.display_gamer_bars(keeper=True, defender=True)
        self.display_human_figures(hum_keeper=kicker.human_keeper)
        self.display_computer_figures(com_keeper=kicker.computer_keeper, com_defender=kicker.computer_defender)
        # self.display_human_figures(hum_keeper=my_human_keeper, hum_defender=my_human_defender)
        # self.display_computer_figures(com_keeper=my_computer_keeper, com_defender=my_computer_defender)
        self.display_goal()
        self.display_score(kicker.get_score())
