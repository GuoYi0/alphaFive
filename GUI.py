# -*- coding: utf-8 -*-
from genData.network import ResNet as Model
import config
import pygame
import os
import numpy as np
from genData.player import Player
import utils
import tensorflow as tf

GRID_WIDTH = 36
WIDTH = (config.board_size + 2) * GRID_WIDTH
HEIGHT = (config.board_size + 2) * GRID_WIDTH
FPS = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HUMAN = 0
AI = 2


def main(trained_ckpt):
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

    # def draw_stone(screen_):
    #     for i in range(config.board_size):
    #         for j in range(config.board_size):
    #             if state[i, j] == 1:
    #                 pygame.draw.circle(screen_, BLACK, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
    #             elif state[i, j] == -1:
    #                 pygame.draw.circle(screen_, WHITE, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
    #             else:
    #                 assert state[i, j] == 0
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

    def visual_update(matrix, file_record, step):
        if step >= len(file_record):
            return False
        else:
            if step % 2 == 0:
                stone = 1
            else:
                stone = -1
            ss = file_record[step]
            matrix[ss[0], ss[1]] = stone
            return True

    players = [HUMAN, AI]  # 0 表示人类玩家，2表示包含network的AI
    idx = int(input("input the fist side, (0 human), (1 AI): "))
    if players[idx] == AI:
        print("AI first")
    else:
        print("Human first")
    game_over = False
    state_str = player.get_init_state()
    board = utils.state_to_board(state_str, config.board_size)
    state = board
    draw_background(screen)
    # draw_stone(screen)
    pygame.display.flip()
    last_action = None
    if players[idx] == AI:
        _, action = player.get_action(state_str, last_action=last_action)
        print("AI's action, ", action)
        board = utils.step(utils.state_to_board(state_str, config.board_size), action)
        state_str = utils.board_to_state(board)
        # player.pruning_tree(board, state_str)  # 走完一步以后，对其他分支进行剪枝，以节约内存
        game_over, value = utils.is_game_over(board, config.goal)
        state = -board
        draw_background(screen)
        draw_stone(screen)
        pygame.display.flip()
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    continue
                pos = event.pos  # 获得的坐标是(x, y)
                if out_of_boundry(pos):
                    continue
                action = (int((pos[1] - GRID_WIDTH) / GRID_WIDTH), int((pos[0] - GRID_WIDTH) / GRID_WIDTH))
                print("Human's action: ", action)
                if state[action[0], action[1]] != 0:
                    continue
                board = utils.step(board, action)  # 人类落子
                last_action = action
                state_str = utils.board_to_state(board)
                # player.pruning_tree(board, state_str)
                game_over, value = utils.is_game_over(board, config.goal)
                state = board
                draw_background(screen)
                draw_stone(screen)
                pygame.display.flip()
                if game_over:
                    break
                _, action = player.get_action(state_str, last_action=last_action)
                last_action = action
                print("AI's action ", action, "============")
                board = utils.step(utils.state_to_board(state_str, config.board_size), action)
                state_str = utils.board_to_state(board)
                player.pruning_tree(board, state_str)  # 走完一步以后，对其他分支进行剪枝，以节约内存
                game_over, value = utils.is_game_over(board, config.goal)
                state = -board
        draw_background(screen)
        draw_stone(screen)
        pygame.display.flip()
    pygame.quit()


def out_of_boundry(pos):
    return pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[1] > HEIGHT - GRID_WIDTH


def print_info(player):
    if player == HUMAN:
        print("human's turn...")
    else:
        print("AI's turn...")


if __name__ == "__main__":
    main(trained_ckpt=config.ckpt_path)
    # main(trained_ckpt="E:\\alphaFive\\five19\\ckpt\\alphaFive-7980")
    # main(trained_ckpt="E:\\alphaFive\\five10\\ckpt\\alphaFive-1140")
