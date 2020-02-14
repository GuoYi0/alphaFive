# -*- coding: utf-8 -*-
import numpy as np
import sys
state = np.array([[0, 1, -1], [0, 0, 1]])

def valid_move(state: np.ndarray):
    return list(tuple(index) for index in np.argwhere(state == 0))




class A(object):
    def __init__(self, a):
        self.a = a

p = A(np.zeros((800, 800, 800)))
q = A(np.zeros((1, 1, 1)))
print(p.__sizeof__())
print(q.__sizeof__())

