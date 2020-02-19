# -*- coding: utf-8 -*-
from network import ResNet as Model
from MCTS import MCTS
import config
import pygame
import os
from gobang import CONTINUE, WON_LOST, DRAW
import numpy as np


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
    # net.restore(trained_ckpt)
    net.restore("E:\\alphaFive\\five1\\ckpt\\alphaFive-180")
    tree = MCTS(config.board_size, net, goal=config.goal)
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
    state = tree.game_process.current_board()


    def draw_stone(screen_):
        for i in range(config.board_size):
            for j in range(config.board_size):
                if state[i, j] == 1:
                    pygame.draw.circle(screen_, BLACK, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
                elif state[i, j] == -1:
                    pygame.draw.circle(screen_, WHITE, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
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
    idx = np.random.randint(2)  # 随机选择一个数，作为最为起始玩家
    terminal = CONTINUE
    if players[idx] == AI:
        print("AI first")
        state, terminal = tree.interact(ai=AI)
        draw_background(screen)
        draw_stone(screen)
        pygame.display.flip()
    else:
        print("human first")
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if terminal != CONTINUE:
                    continue
                pos = event.pos
                if out_of_boundry(pos):
                    continue
                action = (int((pos[0] - GRID_WIDTH) / GRID_WIDTH), int((pos[1] - GRID_WIDTH) / GRID_WIDTH))
                if tree.game_process.is_occupied(action):
                    continue
                state, terminal = tree.interact(action=action, ai=HUMAN)  # 人类落子
                draw_background(screen)
                draw_stone(screen)
                pygame.display.flip()
                if terminal != CONTINUE:
                    break
                state, terminal = tree.interact(ai=AI)  # AI落子
        draw_background(screen)
        draw_stone(screen)
        pygame.display.flip()
    pygame.quit()




    # while True:
    #     clock.tick(FPS)
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             break
    #         elif event.type == pygame.MOUSEBUTTONDOWN:
    #             if terminal == CONTINUE:
    #                 pos = event.pos
    #                 if pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[
    #                     1] > HEIGHT - GRID_WIDTH:
    #                     pass
    #                 else:
    #                     grid = (int((pos[0] - GRID_WIDTH) / GRID_WIDTH), int((pos[1] - GRID_WIDTH) / GRID_WIDTH))
    #                     state, terminal = tree.human_play(action=grid)  # 玩家落子
    #                     draw_background(screen)
    #                     draw_stone(screen)
    #                     pygame.display.flip()
    #                     state, terminal = tree.interact_game(terminal=terminal, state=state, ai=AI)  # AI落子
    #     draw_background(screen)
    #     draw_stone(screen)
    #     pygame.display.flip()
    # pygame.quit()


def out_of_boundry(pos):
    return pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[1] > HEIGHT - GRID_WIDTH

def print_info(player):
    if player == HUMAN:
        print("human's turn...")
    else:
        print("AI's turn...")


if __name__ == "__main__":
    main(trained_ckpt=config.ckpt_path)
