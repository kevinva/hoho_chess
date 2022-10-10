import collections
import random
import json
import os
import torch.nn.functional as F

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
# print(f'{sys.path}')

from torch.utils.data import Dataset, DataLoader

from model.hoho_utils import *
from model.hoho_config import *


class CChessGame:

    def __init__(self, state=INIT_BOARD_STATE, restrict_count=RESTRICT_ROUND_NUM, winner=None, current_player=PLAYER_RED):
        self.state = state
        self.restrict_count = restrict_count
        self.winner = winner
        self.current_player = current_player

        LOGGER.info('CChessGame created!')

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


class ChessDataset(Dataset):

    def __init__(self, data_list=None):
        super(ChessDataset, self).__init__()
        # 注意：需要在这里将list转换为tensor, 否则dataloader取每个batch时，batch_size不会在第0维
        self.data_list = [(exp[0], torch.tensor(exp[1], dtype=torch.float), exp[2]) for exp in data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        experience = self.data_list[idx]
        return experience[0], experience[1], experience[2]

    @staticmethod
    def load_from_dir(dirpath, version=None):
        if not os.path.exists(dirpath):
            return None
            
        all_data_list = list()
        for filename in os.listdir(dirpath):
            if filename.endswith('json') and filename.startswith(REPLAY_BUFFER_FILE_PREFIX):
                if version is None:
                    with open(os.path.join(dirpath, filename), 'r') as f:
                        jsonstr = f.read()
                        data_list = json.loads(jsonstr)
                        all_data_list.extend(data_list)
                else:
                    name = filename.split('.')[0]
                    items = name.split('_')
                    if len(items) == 5:
                        check_version = int(items[4])
                        if version == check_version:
                            with open(os.path.join(dirpath, filename), 'r') as f:
                                jsonstr = f.read()
                                data_list = json.loads(jsonstr)
                                all_data_list.extend(data_list)
                    elif len(items) < 4:
                        # 没有版本后缀的默认为version 0
                        if version == 0:
                            with open(os.path.join(dirpath, filename), 'r') as f:
                                jsonstr = f.read()
                                data_list = json.loads(jsonstr)
                                all_data_list.extend(data_list)

        dataset = ChessDataset(all_data_list)
        return dataset


class Round:

    def __init__(self, round_id):
        self.round_id = round_id
        self.red_steps = list()
        self.black_steps = list()

    def add_red_step(self, state, pi):
        self.red_steps.append((state, pi))
    
    def add_black_step(self, state, pi):
        self.black_steps.append((state, pi))

    def update_winner(self, winner=None):
        if winner is None: 
            reward = 0
            reward_list = list()
            for index, step in enumerate(self.red_steps):
                if index + 1 < len(self.red_steps):
                    next_step = self.red_steps[index + 1]
                    capture_list = check_capture(step[0], next_step[0])
                    if len(capture_list) > 0:
                        step_reward = 0
                        for piece in capture_list:
                            if piece.isupper():  # 红方子被吃
                                step_reward -= chess_value_equal_to_pawn(piece)
                            elif piece.islower():  # 黑方子被吃
                                step_reward += chess_value_equal_to_pawn(piece)

                        reward_list.append(step_reward)
                    else:
                        reward_list.append(reward)
            reward_list.append(reward)

            assert len(rewards) == len(self.red_steps), f'rewards len {len(rewards)} not equal to red_steps len {len(self.red_steps)}'
            
            self.red_steps = [(x[0], x[1], reward_list[i]) for i, x in enumerate(self.red_steps)]

        else:
            reward = 1
            if winner == 'Black':
                reward = -1
                
            total_reward = len(self.red_steps) * reward
            reward_list = list()
            for index, step in enumerate(self.red_steps):
                if index + 1 < len(self.red_steps):
                    next_step = self.red_steps[index + 1]
                    capture_list = check_capture(step[0], next_step[0])
                    if len(capture_list) > 0:
                        step_reward = 0
                        for piece in capture_list:
                            if piece.isupper():  # 红方子被吃
                                step_reward -= chess_value_equal_to_pawn(piece)
                            elif piece.islower():  # 黑方子被吃
                                step_reward += chess_value_equal_to_pawn(piece)

                        if winner  == 'Black':
                            step_reward = -step_reward   # winner为黑方，中间的奖励一般为负（对于红方来说），先转为正数以方便计算softmax

                        reward_list.append(step_reward)
                    else:
                        reward_list.append(reward)
            
            reward_tensor = torch.tensor(reward_list).float()
            reward_ratios = F.softmax(reward_tensor)
            rewards = (reward_ratios * total_reward).tolist()

            if winner == 'Black':
                rewards.append(reward)
            else:
                rewards.append(100.0)   # 红方赢，最后一步强制给100奖励

            assert len(rewards) == len(self.red_steps), f'rewards len {len(rewards)} not equal to red_steps len {len(self.red_steps)}'
            
            self.red_steps = [(x[0], x[1], rewards[i]) for i, x in enumerate(self.red_steps)]
            

    def size(self):
        return len(self.red_steps)


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity=10000, data_list=None):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        if data_list is not None:
            self.buffer.extend(data_list)

    # def add(self, state, pi, z):  
    #     """将数据加入buffer"""
    #     self.buffer.append((state, pi, z))

    def add_round(self, round):
        self.buffer.extend(round.red_steps)

    # def sample(self, batch_size):  
    #     """从buffer中采样数据,数量为batch_size"""

    #     transitions = random.sample(self.buffer, batch_size)
    #     states, pis, zs = zip(*transitions)
    #     return states, pis, zs

    def size(self):  
        """当前buffer中数据的数量"""

        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save(self, expand_data=None):
        filedir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'data')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        
        model_version = 0
        if expand_data is not None:
            model_version = expand_data.get('model_version')
        filename = '{}_{}_{}.json'.format(REPLAY_BUFFER_FILE_PREFIX, int(time.time()), model_version)
        filepath = os.path.join(filedir, filename)
        
        with open(filepath, 'w') as f:
            jsonstr = json.dumps(list(self.buffer))
            f.write(jsonstr)

    @staticmethod
    def load_from_file(filepath):
        data_list = []
        with open(filepath, 'r') as f:
            jsonstr = f.read()
            data_list = json.loads(jsonstr)
        replay_buffer = ReplayBuffer(data_list=data_list)
        return replay_buffer

    @staticmethod
    def load_from_dir(dirpath):
        if not os.path.exists(dirpath):
            return
            
        all_data_list = list()
        for filename in os.listdir(dirpath):
            if filename.endswith('json') and filename.startswith(REPLAY_BUFFER_FILE_PREFIX):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    jsonstr = f.read()
                    data_list = json.loads(jsonstr)
                    all_data_list.extend(data_list)
        replay_buffer = ReplayBuffer(data_list=all_data_list)
        return replay_buffer


if __name__ == '__main__':
    # s = 'ERIOC<VGK1234q24ds'
    # print(('k' in s))

    # q = collections.deque(maxlen=100)
    # q.append(('23', [123, 23], 0))
    # q.append(('24', [123, 23], 0))
    # q.append(('256', [123, 23], 0))
    # print(q)
    # print(list(q))
    # result = json.dumps(list(q))
    # print(result)

    # result_load = json.loads(result)
    # print(result_load)

    # filepath = '../output/data/replay_buffer_1656040190.json'
    # rb = ReplayBuffer.load(filepath)
    # print(rb.buffer)

    # dataset = ChessDataset.load_from_dir('../output/data', version=0)
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # for i, (batch_states, batch_pis, batch_zs) in enumerate(dataloader):
    #     print(len(batch_states), batch_pis.size(), batch_zs)
    #     if i == 0:
    #         break


    # test_list = list()
    # test_list.append((1, '2'))
    # test_list.append((2, '7'))
    # test_list.append((3, '1'))
    # test_list.append((4, '3'))
    # test_list.append((5, '4'))
    # test_list2 = [(x[0], x[1], -1) for x in test_list]
    # print(test_list2)

    rs = [1, 2, 3, 1]
    rt = torch.tensor(rs).float()
    rs = F.softmax(rt) * 10

    print(f'{rs.tolist()}')