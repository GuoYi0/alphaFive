# -*- coding: utf-8 -*-
import numpy as np
state = np.array([[0, 1, -1], [0, 0, 1]])

def valid_move(state: np.ndarray):
    return list(tuple(index) for index in np.argwhere(state == 0))


print(type(int(state[0,0])))
