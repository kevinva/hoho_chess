import numpy as np
from config import C_PUCT
from gameboardv2 import MOVE_2_INDEX, GameBoard

def dirichlet_noise(probas):
    dim = (probas.shape[0])
    new_probas = (1 - P_EPSILON) * probas + P_EPSILON * np.random.dirichlet(np.full(dim, ))


class Node:

    def __init__(self, state, parent=None, proba=None):
        self.state = state
        self.p = proba   # 访问节点的概率，由policy net输出
        self.n = 0       # 节点访问次数
        self.w = 0       # 总行为价值，由value net输出累加, 
        self.q = 0       # 平均价值 w/n
        self.childrens = {}
        self.parent = parent

    def is_leaf(self):
        return len(self.childrens) == 0

    def get_uq_score(self):
        u = C_PUCT * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return u + self.q

    def update(self, value):
        self.n += 1
        self.w = self.w + value
        self.q = self.w / self.n if self.n > 0 else 0

    def select(self):
        return max(self.childrens.values(), key=lambda node: node.get_uq_score())

    def expand(self, legal_actions, all_action_probas):
        nodeInfo = {}
        for action in legal_actions:
            prob = all_action_probas[MOVE_2_INDEX[action]]
            state_new = GameBoard.takeActionOnBoard(action, self.state)
            node = Node(state=state_new, parent=self, proba=prob)
            nodeInfo[action] = node
        self.childrens = nodeInfo

    def backup(self, value):
        node = self
        v = value
        while node != None:
            node.update(v)
            node = self.parent
            v = -v


class MCTS:
    
    def take_smimulation(self, root):
        pass

    



if __name__ == '__main__':
    myl = [11, 23, 3, 55, 23, 7, 20, 29]
    print(max(myl, key=lambda num: 1.0 / num))