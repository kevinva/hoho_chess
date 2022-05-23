import numpy as np
import torch
import threading
import time
import random
from numba import jit
from copy import deepcopy
from config import *
from utils import *


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


class EvaluatorThread(threading.Thread):

    def __init__(self, player, eval_queue, condition_search, condition_eval):
        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.player = player
        self.condition_search = condition_search
        self.condition_eval = condition_eval

    def run(self):
        for sim in range(MCTS_SIMULATION // MCTS_PARAllEL):
            self.condition_search.acquire()
            while len(self.eval_queue) < MCTS_PARALLEL:
                self.condition_search.wait()   # 等待search线程的完成，search下一个节点
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < MCTS_PARALLEL:
                keys = list(self.eval_queue.keys())
                max_len = BATCH_SIZE_EVAL if len(keys) > BATCH_SIZE_EVAL else len(keys)

                states = torch.tensor(np.array(list(self.eval_queue.values()))[0:max_len], dtype=torch.float, device=DEVICE)
                v, probas = self.player.predict(states) # evaluate阶段，输出叶节点的的输赢价值v和下一步走子的概率p

                for idx, i in zip(keys, range(max_len)):
                    del self.eval_queue[idx]
                    self.result_queue[idx] = (probas[i].cpu().data.numpy(), v[i])  # 存放evaluate结果到队列，待expand和update阶段

                self.condition_eval.notifyAll()
            self.condition_eval.release()


class SearchThread(threading.Thread):
    # 用来跑一次模拟

    def __init__(self, mcts, game, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval):
        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.mcts = mcts
        self.game = game
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search

    def run(self):
        game = deepcopy(self.game)
        state = game.state
        current_node = self.mcts.root
        done = False
        
        while not current_node.is_leaf() and not done:
            # select阶段
            idx_selected = select(np.array([[node.q, node.n, node.p] for node in current_node.childrens]))
            current_node = current_node.childrens[idx_selected]
            
            self.lock.acquire()
            current_node.n += 1
            self.lock.release()

            state, _, done = game.step(current_node.move)

        if not done:
            self.condition_search.acquire()

            # Expand and evaluate (Fig. 2b). The leaf node sL is 
            # added to a queue for neural net-work evaluation, (di(p), v) = fθ(di(sL)), 
            # where di is a dihedral reflection or rotation selected uniformly at random from i in [1..8]
            
            # 这时叶节点，把需要evaluat的节点放入队列待EvaluatorThread处理
            # sample_rotation是diheral变换，根据论文，输入网络前需施加给状态s
            self.eval_queue[self.thread_id] = sample_rotation(state, num=1)
            
            self.condition_search.notify()
            self.condition_search.release()

            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()
            
            result = self.result_queue.pop(self.thread_id)
            probas = np.array(result[0])
            v = float(result[1])
            self.condition_eval.release()

            if not current_node.parent:
                # Self-Play: ... Additional exploration is achieved by adding Dirichlet noise 
                # to the prior probabilities in the root node s0, specifically P(s, a) = (1 − ε)pa + εηa, 
                # where η ∼ Dir(0.03) and ε = 0.25; this noise ensures that all moves may be tried, but the search may still overrule bad moves.
                probas = dirichlet_noise(probas)

            valid_moves = game.get_legal_moves()  # hoho: todo
            illegal_moves = np.setdiff1d(np.arange(game.board_size ** 2 + 1), np.array(valid_moves))
            probas[illegal_moves] = 0
            total = np.sum(probas)
            probas /= total

            self.lock.acquire()
            current_node.expand(probas)  # expand阶段

            while current_node.parent:
                current_node.update(v)  # backpropagate阶段
                current_node = current_node.parent
            self.lock.release()


class MCTS:

    def __init__(self):
        self.root = Node()

    def _draw_move(self, action_scores, competitive=False):
        # 生成走子的策略
        if competitive:
            moves = np.where(action_scores == np.max(action_scores))[0]
            move = np.random.choice(moves)
            total = np.sum(action_scores)
            probas = action_scores / total
        else:
            total = np.sum(action_scores)
            probas = action_scores / total
            move = np.random.choice(action_scores.shape[0], p=probas)
        
        return move, probas

    def advance(self, move):
        final_idx = 0
        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == move:
                final_idx = idx
                break
        self.root = self.root.childrens[final_idx]

    def search(self, current_game, player, competitive=False):
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluatorThread(player, eval_queue, result_queue, condition_search, condition_eval)
        evaluator.start()

        threads = []
        for sim in range(MCTS_SIMULATION // MCTS_PARALLEL):
            for idx in range(MCTS_PARALLEL):
                threads.append(SearchThread(self, current_game, eval_queue, result_queue, idx, lock, condition_search, condition_eval))
                threads[-1].start()

            for thread in threads:
                thread.join()
        evaluator.join()

        # visit count 向量
        action_scores = np.zeros((current_game.board_size ** 2 + 1,))  # hoho:改棋盘大小
        for node in self.root.childrens:
            action_scores[node.move] = node.n

        # 选最好的走法
        final_move, final_probas = self._draw_move(action_scores, competitive=competitive)

        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == final_move:
                break
        self.root = self.root.childrens[idx]

        return final_probas, final_move

