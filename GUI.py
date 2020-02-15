# -*- coding: utf-8 -*-
from network import ResNet as Model
from MCTS import MCTS
import config
import pygame
import os
from gobang import CONTINUE, WON_LOST, DRAW

GRID_WIDTH = 36
WIDTH = (config.board_size + 2) * GRID_WIDTH
HEIGHT = (config.board_size + 2) * GRID_WIDTH
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def main(trained_ckpt):
    net = Model(config.board_size)
    net.restore(trained_ckpt)
    tree = MCTS(config.board_size, net, goal=config.goal)
    state, terminal, _ = tree.interact_game_init()
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

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if terminal == CONTINUE:
                    pos = event.pos
                    if pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[
                        1] > HEIGHT - GRID_WIDTH:
                        pass
                    else:
                        grid = (int((pos[0] - GRID_WIDTH) / GRID_WIDTH), int((pos[1] - GRID_WIDTH) / GRID_WIDTH))
                        state, terminal = tree.interact_game(action=grid)  # 玩家落子
                        draw_background(screen)
                        draw_stone(screen)
                        pygame.display.flip()
                        state, terminal, _ = tree.interact_game_ai(action=grid, terminal=terminal, state=state)  # AI落子
        draw_background(screen)
        draw_stone(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main(trained_ckpt=config.ckpt_path)
