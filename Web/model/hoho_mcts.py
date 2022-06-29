import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
# print(f'{sys.path}')

import math
import numpy as np
import threading
from copy import deepcopy
from collections import OrderedDict
from model.hoho_config import *
from model.hoho_utils import *

SEARCH_THREAD_GAME_DONE = 'search thread game done'


def dirichlet_noise(probas):
    dim = (probas.shape[0])
    new_probas = (1 - NOISE_EPSILON) * probas + NOISE_EPSILON * np.random.dirichlet(np.full(dim, NOISE_ALPHA))
    return new_probas


class Node:

    def __init__(self, to_state=INIT_BOARD_STATE, action=None, parent=None, proba=None, player=PLAYER_RED):
        self.to_state = to_state
        self.action = action  # 导致该state的action
        self.childrens = []
        self.parent = parent
        self.player = player # 当前的玩家
        self.P = proba   # 访问节点的概率，由policy net输出
        self.N = 0       # 节点访问次数
        self.W = 0       # 总行为价值，由value net输出累加, 
        self.Q = 0       # 平均价值 w/n

    def get_uq_score(self):
        U = C_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return U + self.Q

    def is_leaf(self):
        return len(self.childrens) == 0

    def select(self):
        return max(self.childrens, key=lambda node: node.get_uq_score())

    def expand(self, all_action_probas, legal_actions):
        nodes = []
        node_player = PLAYER_RED
        if self.player == PLAYER_RED:  # 当前节点玩家与其子节点玩家互异
            node_player = PLAYER_BLACK

        for action in legal_actions:
            prob = all_action_probas[ACTIONS_2_INDEX[action]]
            to_state_new = do_action_on_board(self.to_state, action)
            node = Node(to_state=to_state_new, action=action, parent=self, proba=prob, player=node_player)
            nodes.append(node)
        self.childrens = nodes

    def backup(self, value):
        v = value
        node = self
        while node.parent is not None:
            # node.N += 1   # virtual loss加了，这里就不加了
            node.W = node.W + v
            node.Q = node.W / node.N if node.N > 0 else 0
            v = -v   # 本节点与父节点为不同玩家，所以其价值增长互反
            node = node.parent


    def node_desc(self):
        node_info = dict()
        node_info['action'] = self.action
        node_info['to_state'] = self.to_state
        node_info['N'] = self.N
        node_info['W'] = self.W
        node_info['Q'] = self.Q
        node_info['P'] = self.P
        node_info['children'] = dict()
        for child in self.childrens:
            child_info = child.node_desc()
            node_info['children'][child.action] = child_info
        
        return node_info



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
        state = game.state
        current_node = self.root_node
        done = False
        
        assert state == current_node.to_state, "{} game's state should be equal to node's state".format(LOG_TAG_MCTS)

        while not current_node.is_leaf() and not done:
            # 一直select到叶节点
            current_node = current_node.select()

            # 加上virtual loss
            self.lock.acquire()
            current_node.N += 1
            self.lock.release()

            next_state, _, done = game.step(current_node.action)
            state = next_state

        if not done:  # 找到叶节点但游戏还没结束
            self.condition_search.acquire()
            if current_node.player == PLAYER_BLACK:
                # 如果当前节点是黑方待下棋，则将棋盘翻转，让黑方以红方的视角走子（self play）
                # print('flip the board!')
                state = flip_board(state)
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
                # print('should flip the action probability')
                probas = flip_action_probas(probas)
            value = float(result[1])
            self.condition_eval.release()

            if not current_node.parent:
                probas = dirichlet_noise(probas)

            legal_actions = get_legal_actions(current_node.to_state, current_node.player)
            
            self.lock.acquire()

            # 叶节点expand
            current_node.expand(probas, legal_actions)

            # 从叶节点backup
            current_node.backup(-value)  # 注意这里value要变成相反数，因为神经网络输出的是当前状态的价值，即当前玩家在当前棋局下的胜负价值，而当前节点的W值是上一轮玩家进行动作后的价值，自然是相反了
            
            self.lock.release()

        else: 
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
            current_node.backup(value)
            self.lock.release()

            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = SEARCH_THREAD_GAME_DONE
            self.condition_search.notify()
            self.condition_search.release()

            # print(f'{LOG_TAG_MCTS} {SEARCH_THREAD_GAME_DONE}')


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
            # print(f'evaluate thread: {simulation}')
            self.condition_search.acquire()
            while len(self.eval_queue) < MCTS_THREAD_NUM:   # 需等同一批搜索线程都入队列，才开始下一步
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            batch_result_count = MCTS_THREAD_NUM
            while len(self.result_queue) < batch_result_count:
                thread_ids = list(self.eval_queue.keys())
                planes = list()
                eval_thread_ids = list()
                for tid in thread_ids:
                    state = self.eval_queue[tid]
                    if state != SEARCH_THREAD_GAME_DONE:
                        board_state = state
                        plane = convert_board_to_tensor(board_state)
                        planes.append(plane)
                        eval_thread_ids.append(tid)
                    else:
                        del self.eval_queue[tid]
                        batch_result_count = batch_result_count - 1
                
                if len(planes) > 0:
                    batch_states = torch.stack(planes, dim=0).to(DEVICE)
                    batch_probas, batch_values = self.agent.predict(batch_states)

                    for idx, tid in enumerate(eval_thread_ids):
                        self.result_queue[tid] = (batch_probas[idx].to(torch.device('cpu')).detach().numpy(), batch_values[idx])
                        del self.eval_queue[tid]
                    self.condition_eval.notifyAll()
            self.condition_eval.release()


class MCTS:

    def __init__(self, start_player=PLAYER_RED, start_state=INIT_BOARD_STATE):
        self.root = Node(to_state=start_state, player=start_player)

        print(f'{LOG_TAG_MCTS} MCTS created!')
    
    def take_simulation(self, agent, game, update_root=True):
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
                searcher = SearchThread(self.root, game, eval_queue, result_queue, idx, lock, condition_search, condition_eval)
                searcher.start()
                search_threads.append(searcher)
            for thread in search_threads:
                thread.join()

            # if DEBUG:
            #     print(f'{LOG_TAG_MCTS} simulation: {simulation}')

        evaluator.join()
    
        # 模拟走子之后，生成走子策略
        action_scores = np.zeros((ACTION_DIM,))
        for node in self.root.childrens:
            action_scores[ACTIONS_2_INDEX[node.action]] = np.power(node.N, 1 / POLICY_TEMPERATURE)
        
        total = np.sum(action_scores)
        if total == 0:
            total += 1e-3
        pi = action_scores / total
        final_action_idx = np.random.choice(action_scores.shape[0], p=pi)

        # 替换为新的根节点
        final_action = INDEXS_2_ACTION[final_action_idx]
        if update_root:
            self.update_root_with_action(final_action)

        return pi, final_action

    def update_root_with_action(self, action):
        """让action对应的子节点成为新的根节点"""

        found_idx = -1
        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].action == action:
                found_idx = idx
                break

        if found_idx >= 0 and found_idx < len(self.root.childrens):
            self.root = self.root.childrens[found_idx]
            self.root.parent = None
        else:
            print(f'{LOG_TAG_MCTS} Update tree root error! found_idx={found_idx}')

    def is_current_root_expanded(self):
        return len(self.root.childrens) > 0

if __name__ == '__main__':
    myl = [11, 23, 3, 55, 23, 7, 20, 29]
    print(np.random.choice(len(myl)))