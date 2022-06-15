import numpy as np
import threading
from copy import deepcopy
from collections import OrderedDict
from model.config import *
from model.gameboard import *


def dirichlet_noise(probas):
    dim = (probas.shape[0])
    new_probas = (1 - NOISE_EPSILON) * probas + NOISE_EPSILON * np.random.dirichlet(np.full(dim, NOISE_ALPHA))
    return new_probas


class Node:

    def __init__(self, state=INIT_BOARD_STATE, action=None, parent=None, proba=None, player=PLAYER_RED):
        self.state = state
        self.action = action  # 导致该state的action
        self.P = proba   # 访问节点的概率，由policy net输出
        self.N = 0       # 节点访问次数
        self.W = 0       # 总行为价值，由value net输出累加, 
        self.Q = 0       # 平均价值 w/n
        self.childrens = []
        self.parent = parent
        self.player = player

    def get_uq_score(self):
        U = C_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return U + self.Q

    def is_leaf(self):
        return len(self.childrens) == 0

    def select(self):
        return max(self.childrens, key=lambda node: node.get_uq_score())

    def expand(self, legal_actions, all_action_probas):
        nodes = []
        node_player = PLAYER_RED
        if self.player == PLAYER_RED:
            node_player = PLAYER_BLACK

        for action in legal_actions:
            prob = all_action_probas[ACTIONS_2_INDEX[action]]
            state_new = GameBoard.takeActionOnBoard(action, self.state)
            node = Node(state=state_new, action=action, parent=self, proba=prob, player=node_player)
            nodes.append(node)
        self.childrens = nodes

    def backupToRoot(self, value):
        node = self
        v = value
        while node != None:
            node.N += 1
            node.W = node.W + v
            node.Q = node.W / node.N if node.N > 0 else 0
            node = self.parent
            v = -v

    def backup(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N if self.N > 0 else 0



class SearchThread(threading.Thread):

    def __init__(self, root_node, game, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval):
        super(SearchThread, self).__init__()

        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.root_node = root_node
        self.game = game
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search

    def run(self):
        game = deepcopy(self.game)
        state = game.state   # hoho_todo: game的定义
        current_node = self.root_node
        done = False

        while not current_node.is_lead() and not done:
            current_node = current_node.select()

            # 加上virtual loss
            self.lock.acquire()
            current_node.N += VIRTUAL_LOSS
            current_node.W -= VIRTUAL_LOSS
            self.lock.release()

            state, _, done = game.step(current_node.action)  # hoho_todo： game的定义

        if done:
            value = 0.0
            if game.winner == PLAYER_RED:
                if current_node.player == PLAYER_BLACK:
                    value = 1.0
                elif current_node.player == PLAYER_RED:
                    value = -1.0
            elif game.winner == PLAYER_BLACK:
                if current_node.player == PLAYER_RED:
                    value = 1.0
                elif current_node.player == PLAYER_BLACK:
                    value = -1.0
            
            self.lock.acquire()
            node_tmp = current_node
            v = value
            while node_tmp != None:
                # backup之前把之前加的virtual loss撤销掉
                node_tmp.N -= VIRTUAL_LOSS
                node_tmp.W += VIRTUAL_LOSS
                node_tmp.backup(v)
                node_tmp = node_tmp.parent
                v = -v
            self.lock.acquire()
            
            # TODO
        else:
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = state   # hoho_todo: 按paper，这里可考虑增加一个dihedral transformation
            self.condition_search.notify()
            self.condition_search.release()

            # 阻塞自己，等待evaluate线程完成并唤醒自己
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            result = self.result_queue.pop(self.thread_id)
            probas = np.array(result[0])
            v = float(result[1])
            self.condition_eval.release()

            if not current_node.parent:
                probas = dirichlet_noise(probas)
            
            legal_actions = GameBoard.get_legal_actions()

            # TODO



class EvaluateThread(threading.Thread):

    def __init__(self, player, eval_queue, result_queue, condition_search, condition_eval):
        super(EvaluateThread, self).__init__()



class MCTS:

    def __init__(self):
        self.root = Node()
    
    def take_simulation(self, player, current_game):
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluateThread()

    
    




if __name__ == '__main__':
    myl = [11, 23, 3, 55, 23, 7, 20, 29]
    print(max(myl, key=lambda num: 1.0 / num))