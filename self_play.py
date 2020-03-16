# -*- coding: utf-8 -*-
from genData.network import ResNet as Model
import config
import pygame
import os
import numpy as np
from genData.player import Player
import utils
import cv2
import imageio


GRID_WIDTH = 36
WIDTH = (config.board_size + 2) * GRID_WIDTH
HEIGHT = (config.board_size + 2) * GRID_WIDTH
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HUMAN = 0
AI = 2


def main(trained_ckpt):
    print(config.simulation_per_step)
    net = Model(config.board_size)
    player = Player(config, training=False, pv_fn=net.eval)
    net.restore(trained_ckpt)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("五子棋")
    clock = pygame.time.Clock()
    base_folder = os.path.dirname(__file__)
    img_folder = os.path.join(base_folder, 'images')
    background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()
    background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
    back_rect = background.get_rect()
    running = True
    frames = []
    def draw_stone(screen_):
        for i in range(config.board_size):
            for j in range(config.board_size):
                if state[i, j] == 1:
                    pygame.draw.circle(screen_, BLACK, (int((j + 1.5) * GRID_WIDTH), int((i + 1.5) * GRID_WIDTH)), 16)
                elif state[i, j] == -1:
                    pygame.draw.circle(screen_, WHITE, (int((j + 1.5) * GRID_WIDTH), int((i + 1.5) * GRID_WIDTH)), 16)
                else:
                    assert state[i, j] == 0

    def draw_background(surf):
        screen.blit(background, back_rect)
        rect_lines = [
            ((GRID_WIDTH, GRID_WIDTH), (GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((GRID_WIDTH, GRID_WIDTH), (WIDTH - GRID_WIDTH, GRID_WIDTH)),
            ((GRID_WIDTH, HEIGHT - GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((WIDTH - GRID_WIDTH, GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
        ]
        for line in rect_lines:
            pygame.draw.line(surf, BLACK, line[0], line[1], 2)

        for i in range(config.board_size):
            pygame.draw.line(surf, BLACK,
                             (GRID_WIDTH * (2 + i), GRID_WIDTH),
                             (GRID_WIDTH * (2 + i), HEIGHT - GRID_WIDTH))
            pygame.draw.line(surf, BLACK,
                             (GRID_WIDTH, GRID_WIDTH * (2 + i)),
                             (HEIGHT - GRID_WIDTH, GRID_WIDTH * (2 + i)))

        circle_center = [
            (GRID_WIDTH * 4, GRID_WIDTH * 4),
            (WIDTH - GRID_WIDTH * 4, GRID_WIDTH * 4),
            (WIDTH - GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
            (GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
        ]
        for cc in circle_center:
            pygame.draw.circle(surf, BLACK, cc, 5)

    game_over = False
    state_str = player.get_init_state()
    board = utils.state_to_board(state_str, config.board_size)
    state = board
    draw_background(screen)
    pygame.display.flip()
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    frames.append(cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5))
    turn = 0
    i = 0
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        action = None
        if not game_over:
            _, action = player.get_action(state_str, last_action=action)
            board = utils.step(utils.state_to_board(state_str, config.board_size), action)
            state_str = utils.board_to_state(board)
            # player.pruning_tree(board, state_str)  # 走完一步以后，对其他分支进行剪枝，以节约内存
            game_over, value = utils.is_game_over(board, config.goal)
            if turn %2 ==1:
                state = board
            else:
                state = -board
            turn += 1
            draw_background(screen)
            draw_stone(screen)
            pygame.display.flip()
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            frames.append(cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5))

        # draw_background(screen)
        # draw_stone(screen)
        # pygame.display.flip()
        if game_over:
            i += 1
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            frames.append(cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5))
            if i >= 3:  # 最终保留三帧
                break

    pygame.quit()
    print("game finished, start to write to gif.")
    gif = imageio.mimsave("tmp/five.gif", frames,'GIF', duration=0.8)
    print("done!")


if __name__ == "__main__":
    main(trained_ckpt=config.ckpt_path)
