import collections
import random
from model.hoho_utils import *
from model.hoho_config import *



class CChessGame:

    def __init__(self, state=INIT_BOARD_STATE, restrict_count=RESTRICT_ROUND_NUM, winner=None, current_player=PLAYER_RED):
        self.state = state
        self.restrict_count = restrict_count
        self.winner = winner
        self.current_player = current_player

    def step(self, action):
        state_new = do_action_on_board(self.state, action)
        z = 0
        done = False
        if 'K' not in state_new:
            z = -1
            done = True
            self.winner = PLAYER_BLACK
        elif 'k' not in state_new:
            z = 1
            done = True
            self.winner = PLAYER_RED
        else:
            # hoho_todo: 增加restrict_count的逻辑
            pass

        self.state = state_new
        return state_new, z, done

    def reset(self):
        self.state = INIT_BOARD_STATE
        self.restrict_count = RESTRICT_ROUND_NUM
        self.winner = None
        self.current_player = PLAYER_RED



class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, pi, z):  
        """将数据加入buffer"""

        self.buffer.append((state, pi, z))

    def sample(self, batch_size):  
        """从buffer中采样数据,数量为batch_size"""

        transitions = random.sample(self.buffer, batch_size)
        states, pis, zs = zip(*transitions)
        return states, pis, zs

    def size(self):  
        """当前buffer中数据的数量"""

        return len(self.buffer)



if __name__ == '__main__':
    s = 'ERIOC<VGK1234q24ds'
    print(('k' in s))