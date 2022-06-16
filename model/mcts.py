import numpy as np
import threading
from copy import deepcopy
from collections import OrderedDict
from config import *
from gameboard import *


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
            state_new = GameBoard.do_action_on_board(action, self.state)
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
            self.lock.release()

        else:  # 找到叶节点但游戏还没结束
            self.condition_search.acquire()

            # 如果当前节点是黑方待下棋，则将棋盘翻转，让黑方以红方的视角走子（self play）
            if current_node.player == PLAYER_BLACK:
                print('flip the board!')
                state = GameBoard.flip_board(state)
            self.eval_queue[self.thread_id] = state   # hoho_todo: 按paper，这里可考虑增加一个dihedral transformation
            self.condition_search.notify()
            self.condition_search.release()

            # 阻塞自己，等待evaluate线程完成并唤醒自己
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            result = self.result_queue.pop(self.thread_id)
            probas = np.array(result[0])
            
            # 因为之前对黑方进行棋盘翻转，所以网络输出的是红方视角的走子方式，要对该走子方式再翻转过来才是黑方真正的走子方式
            if current_node.player == PLAYER_BLACK:
                print('should flip the action probability')
                probas = flip_action_probas(probas)
            value = float(result[1])
            self.condition_eval.release()

            if not current_node.parent:
                probas = dirichlet_noise(probas)
            
            legal_actions = GameBoard.get_legal_actions(current_node.state, current_node.player)
            
            self.lock.acquire()
            # 叶节点expand
            current_node.expand(legal_actions, probas)

            # backup
            node_tmp = current_node
            v = value
            while node_tmp != None:
                node_tmp.N -= VIRTUAL_LOSS
                node_tmp.W += VIRTUAL_LOSS
                node_tmp.backup(v)
                node_tmp = node_tmp.parent
                v = -v
            self.lock.release()


class EvaluateThread(threading.Thread):

    def __init__(self, agent, eval_queue, result_queue, condition_search, condition_eval):
        super(EvaluateThread, self).__init__()

        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.agent = agent
        self.condition_search = condition_search
        self.condition_eval = condition_eval

    def run(self):
        for simulation in range(MCTS_SIMULATION_NUM // MCTS_THREAD_NUM):
            self.condition_search.acquire()
            while len(self.eval_queue) < MCTS_THREAD_NUM:   # 需等同一批搜索线程都入队列，才开始下一步
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < MCTS_THREAD_NUM:
                thread_ids = list(self.eval_queue.keys())
                planes = list()
                for key in thread_ids:
                    plane = GameBoard.convert_board_to_tensor(self.eval_queue[key])
                    planes.append(plane)
                
                batch_states = torch.stack(planes, dim=0).to(DEVICE)
                batch_probas, batch_values = self.agent.predict(batch_states)

                for idx, key in enumerate(thread_ids):
                    self.result_queue[key] = (batch_probas[idx].to(torch.device('cpu')).numpy(), batch_values)
                    del self.eval_queue[key]
                self.condition_eval.notifyAll()
            self.condition_eval.release()


class MCTS:

    def __init__(self):
        self.root = Node()
    
    def take_simulation(self, agent, game):
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluateThread(agent, eval_queue, result_queue, condition_search, condition_eval)
        evaluator.start()

        search_threads = []
        for simulation in range(MCTS_SIMULATION_NUM // MCTS_THREAD_NUM):
            for idx in range(MCTS_THREAD_NUM):
                searcher = SearchThread(self.root, agent, eval_queue, result_queue, idx, lock, condition_search, condition_eval)
                searcher.start()
                search_threads.append(searcher)
            for thread in search_threads:
                thread.join()
        evaluator.join()
    
        # 模拟走子之后，生成走子策略
        action_scores = np.zeros((ACTION_DIM,))
        for node in self.root.childrens:
            action_scores[ACTIONS_2_INDEX[node.action]] = node.N
        total = np.sum(action_scores)
        pi = action_scores / total
        final_action_idx = np.random.choice(action_scores.shape[0], p=pi)

        # 替换新的根节点
        final_idx = -1
        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].action == INDEXS_2_ACTION[final_action_idx]:
                final_idx = idx
                break
        self.root = self.root.childrens[final_idx]

        return pi, final_action_idx



if __name__ == '__main__':
    myl = [11, 23, 3, 55, 23, 7, 20, 29]
    print(np.random.choice(len(myl)))