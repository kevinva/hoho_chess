import numpy as np
import torch
import threading
import time
import random
from numba import jit
from config import *

@jit(nopython=True)
def select(nodes, c_puct=C_PUCT):
    total_count = 0
    for i in range(nodes.shape[0]):
        total_count += nodes[i][1]  # 矩阵nodes的每一行代表一个节点，每个节点有4维，分别是：P、N、W、Q四个值 

    action_scores = np.zeros(nodes.shape[0])
    for i in range(nodes.shape[0]):
        Q = nodes[i][0]
        N = nodes[i][1]
        p = nodes[i][2]

        action_scores[i] = Q + c_puct * p * (np.sqrt(total_count) / (1 + N))
    action_indexs = np.where(action_scores == np.max(action_scores))[0]
    if action_indexs.shape[0] > 0:
        return np.random.choice(action_indexs)
    else:
        return action_indexs[0]


def dirichlet_noise(probas):
    dim = (probas.shape[0])
    new_probas = (1 - P_EPSILON) * probas + P_EPSILON * np.random.dirichlet(np.full(dim, ))

class Node:

    def __init__(self, parent=None, proba=None, move=None):
        self.p = proba   # 访问节点的概率，由policy net输出
        self.n = 0       # 节点访问次数
        self.w = 0       # 总行为价值，由value net输出
        self.q = 0       # 平均价值 w/n
        self.childrens = []
        self.parent = parent
        self.move = move

    def is_leaf(self):
        return len(self.childrens) == 0

    def update(self, v):
        self.w = self.w + v
        self.q = self.w / self.n if self.n > 0 else 0

    def expand(self, probas):
        self.childrens = [Node(parent=self, proba=probas[idx], move=idx) for idx in range(probas.shape[0]) if probas[idx] > 0]


class MCTS:

    def __init__(self)