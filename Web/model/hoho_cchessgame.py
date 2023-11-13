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
        state_new = do_action_on_board(self.state, action)  # 注意：这里state以红方视觉为准，黑方的action不用flip
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
                        trajectories = json.loads(jsonstr)
                        for trajectory in trajectories:
                            all_data_list.extend(trajectory)
                else:
                    name = filename.split('.')[0]
                    items = name.split('_')  
                    if len(items) == 3:
                        check_version = int(items[2])
                        if version == check_version:
                            with open(os.path.join(dirpath, filename), 'r') as f:
                                jsonstr = f.read()
                                trajectories = json.loads(jsonstr)
                                for trajectory in trajectories:
                                    all_data_list.extend(trajectory)
                    elif len(items) < 4:
                        # 没有版本后缀的默认为version 0
                        if version == 0:
                            with open(os.path.join(dirpath, filename), 'r') as f:
                                jsonstr = f.read()
                                trajectories = json.loads(jsonstr)
                                for trajectory in trajectories:
                                    all_data_list.extend(trajectory)

        dataset = ChessDataset(all_data_list)
        return dataset


class Round:

    def __init__(self, round_id):
        self.round_id = round_id
        self.red_steps = list()       # 只有红方的steps
        self.black_steps = list()     # 只有黑方的steps
        self.all_step_list = list()   # 包含红黑双方的steps

    def add_red_step(self, current_state, pi, action_taken, mid_state, r, done):
        self.red_steps.append((current_state, pi, action_taken, mid_state, r, done))
    
    def add_black_step(self, current_state, pi, action_taken, mid_state, r, done):
        self.black_steps.append((current_state, pi, action_taken, mid_state, r, done))

    # 检查每步吃子的情况，更新reward，每到局终时需要调一下(这里也进行奖励重分配)
    def update_winner(self, winner = None):

        # LOGGER.info(f"winner: {winner}, black: {len(self.black_steps)}, red: {len(self.red_steps)}")

        chapture_reward_list = list()
        capture_list = list()
        next_state_list = list()
        for index, step in enumerate(self.red_steps):
            if index + 1 < len(self.red_steps):
                next_step = self.red_steps[index + 1]

                next_state_list.append(next_step[0])

                step_captures = check_capture(step[0], next_step[0])  # 计算当前棋局下，走当前步后多吃子情况
                capture_list.append(step_captures)

                if len(step_captures) > 0:  # 吃子数据是个列表是因为对于其中一方多steps列表，每个step之间是包含红方一步和黑方一步的，所以step之间最多可以有两步吃子
                    step_reward = 0
                    for piece in step_captures:
                        if piece.isupper():  # 红方子被吃
                            step_reward -= chess_value_equal_to_pawn(piece)
                        elif piece.islower():  # 黑方子被吃
                            step_reward += chess_value_equal_to_pawn(piece)

                    chapture_reward_list.append(step_reward)
                else:
                    chapture_reward_list.append(0) # 没有吃子，reward为0

        # 最后step的处理
        current_state = None
        final_state = None
        if len(self.red_steps) > len(self.black_steps):  # 最终步结束于红方
            red_last_step = self.red_steps[-1]
            current_state = red_last_step[0]
            final_state = red_last_step[3]
            next_state_list.append(final_state)
        else:                                             # 最终步结束于黑方
            current_state = self.red_steps[-1][0]
            black_last_step = self.black_steps[-1]
            final_state = flip_board(black_last_step[3]) # 注意：黑方的当前棋局需要翻转为红方视觉
            next_state_list.append(final_state)

        step_captures = check_capture(current_state, final_state)
        capture_list.append(step_captures)

        # LOGGER.info(f"current_state: {current_state}, final_state: {final_state}， step_captures: {step_captures}")

        if len(step_captures) > 0:  # 吃子数据是个列表是因为对于其中一方多steps列表，每个step之间是包含红方一步和黑方一步的，所以step之间最多可以有两步吃子
            step_reward = 0
            for piece in step_captures:
                if piece.isupper():  # 红方子被吃
                    step_reward -= chess_value_equal_to_pawn(piece)
                elif piece.islower():  # 黑方子被吃
                    step_reward += chess_value_equal_to_pawn(piece)

            chapture_reward_list.append(step_reward)
        else:
            chapture_reward_list.append(0) # 没有吃子，reward为0

        assert len(chapture_reward_list) == len(self.red_steps), f"chapture rewards len '{len(chapture_reward_list)}' not equal to red_steps len '{len(self.red_steps)}'"
        assert len(capture_list) == len(self.red_steps), f"capture_list len '{len(capture_list)}' should be equal to red_steps len '{len(self.red_steps)}'"
        assert len(next_state_list) == len(self.red_steps), f"next_state_list len '{len(next_state_list)}' should be equal to red_steps len '{len(self.red_steps)}'"

        # x[0]: current_state, 
        # x[1]: pi, 
        # x[2]: action_taken, 
        # x[4]: reward_raw，以红方视觉：赢为1，输为-1，其他为0
        # x[5]: done
        self.red_steps = [(x[0], x[1], x[2], next_state_list[i], x[4], x[5], capture_list[i], chapture_reward_list[i]) for i, x in enumerate(self.red_steps)]

        # 纠正数据
        final_red_step = list(self.red_steps[-1])
        final_capture_list = final_red_step[6]
        win = 0
        if "K" in final_capture_list:
            win = -1
        elif "k" in final_capture_list:
            win = 1
        final_red_step[5] = True  # done
        final_red_step[4] = win
        
        del(self.red_steps[-1])
        self.red_steps.append(tuple(final_red_step))

        self.red_steps = Round.redistribute_reward(self.red_steps)

    # 包含红黑双方的数据
    def update_winner_v2(self, winner = None):
        all_steps = []
        all_len = len(self.red_steps) + len(self.black_steps)
        r_idx = 0  # 红方索引
        b_idx = 0  # 黑方索引
        chapture_reward_list = list()
        capture_list = list()
        player_list = list()
        while len(all_steps) < all_len:
            ###### 统计红方
            red_step = self.red_steps[r_idx]

            red_step_captures = check_capture(red_step[0], red_step[3])  # 计算当前棋局下，走当前步后吃子情况
            capture_list.append(red_step_captures)

            red_step_reward = 0
            for piece in red_step_captures:
                if piece.isupper():  # 红方子被吃
                    red_step_reward -= chess_value_equal_to_pawn(piece)
                elif piece.islower():  # 红方吃子  (只有这种情况)
                    red_step_reward += chess_value_equal_to_pawn(piece)
            chapture_reward_list.append(red_step_reward)

            player_list.append("r")
            all_steps.append(red_step)
            r_idx += 1
            
            
            if len(all_steps) == all_len:
                # 如果红方获胜应该会走到这一步
                break

            ###### 统计黑方
            black_step = self.black_steps[b_idx]

            black_step_captures = check_capture(black_step[0], black_step[3])  # 计算当前棋局下，走当前步后吃子情况
            capture_list.append(black_step_captures)

            black_step_reward = 0
            for piece in black_step_captures:
                if piece.isupper():  # 红方子被吃 （只有这种情况）
                    black_step_reward -= chess_value_equal_to_pawn(piece)
                elif piece.islower():  # 红方吃子
                    black_step_reward += chess_value_equal_to_pawn(piece)
            chapture_reward_list.append(black_step_reward)

            player_list.append("b")
            all_steps.append(black_step)
            b_idx += 1

        assert len(chapture_reward_list) == len(all_steps), f"chapture rewards len '{len(chapture_reward_list)}' not equal to all_steps len '{len(all_steps)}'"
        assert len(capture_list) == len(all_steps), f"capture_list len '{len(capture_list)}' should be equal to all_steps len '{len(all_steps)}'"

        all_steps_new = [(x[0], x[1], x[2], x[3], x[4], x[5], capture_list[i], chapture_reward_list[i], player_list[i]) for i, x in enumerate(all_steps)]


        # 纠正数据
        final_step = list(all_steps_new[-1])
        final_capture_list = final_step[6]
        win = 0
        if "K" in final_capture_list:
            win = -1
        elif "k" in final_capture_list:
            win = 1
        final_step[5] = True  # done
        final_step[4] = win
        
        del(all_steps_new[-1])
        all_steps_new.append(tuple(final_step))
        all_steps_new = Round.redistribute_reward_v2(all_steps_new)
        self.all_step_list = all_steps_new


    # 奖励重塑
    @staticmethod
    def redistribute_reward(episode_step_list):
        final_step = episode_step_list[-1]
        final_reward = final_step[4]
        raw_rewards = np.array([final_reward] * len(episode_step_list))
        step_count = len(episode_step_list)
        # 构造奖励衰减矩阵
        reward_mat = np.zeros((step_count, step_count))

        # print(f"step_count: {step_count}")

        for t in range(step_count):
            step = episode_step_list[t]
            chapture_reward = step[7]

            left_bound = max(0, t - RER_WINDOW_SIZE)
            right_bound = min(t + RER_WINDOW_SIZE, step_count - 1)

            # print(f"t: {t}, left - right: {left_bound} - {right_bound}")

            reward_mat[t][t] = chapture_reward * RER_LAMBDA

            # 向前衰减
            for i, val in enumerate(range(t, left_bound, -1)):
                reward_mat[t][val - 1] = chapture_reward * pow(RER_LAMBDA, i + 1)

            # 向后衰减
            for i, val in enumerate(range(t, right_bound)):
                reward_mat[t][val + 1] = chapture_reward * pow(RER_LAMBDA, i + 1)


        # print(f"reward_mat: {reward_mat}")

        # 按列相加得出每一步的附加奖励
        addition_rewards = np.sum(reward_mat, axis = 0)

        # print(f"addition_rewards: {addition_rewards}")

        assert raw_rewards.shape[0] == addition_rewards.shape[0]
        assert addition_rewards.shape[0] == len(episode_step_list)

        final_rewards = raw_rewards + addition_rewards
        result_steps = [(episode_info[0], episode_info[1], episode_info[2], episode_info[3], episode_info[4], episode_info[5], episode_info[6], episode_info[7], episode_info[8], final_reward) for episode_info, final_reward in zip(episode_step_list, final_rewards)]
        
        return result_steps
    
    @staticmethod
    def redistribute_reward_v2(episode_step_list):
        step_count = len(episode_step_list)
        raw_rewards = np.zeros((step_count,))
        final_step = episode_step_list[-1]
        final_reward = final_step[4]
        if final_reward != 0:
            r = final_reward
            for i in range(raw_rewards.shape[0]):
                raw_rewards[step_count - 1 - i] = r
                r = -r  # 红黑方交替奖励

        # 构造奖励衰减矩阵
        reward_mat = np.zeros((step_count, step_count))

        # print(f"step_count: {step_count}")

        for t in range(step_count):
            step = episode_step_list[t]
            chapture_reward = step[7]

            left_bound = max(0, t - RER_ALL_STEP_WINDOW_SIZE)
            # right_bound = min(t + RER_ALL_STEP_WINDOW_SIZE, step_count - 1)  # 向后衰减需要计算右边界

            # print(f"t: {t}, left - right: {left_bound} - {right_bound}")

            reward_mat[t][t] = chapture_reward * RER_LAMBDA

            # 向前衰减
            for i, val in enumerate(range(t, left_bound, -1)):
                reward_mat[t][val - 1] = chapture_reward * pow(RER_LAMBDA, i + 1)

            # # 向后衰减
            # for i, val in enumerate(range(t, right_bound)):
            #     reward_mat[t][val + 1] = chapture_reward * pow(RER_LAMBDA, i + 1)


        # print(f"reward_mat: {reward_mat}")

        # 按列相加得出每一步的附加奖励
        addition_rewards = np.sum(reward_mat, axis = 0)

        # print(f"addition_rewards: {addition_rewards}")

        assert raw_rewards.shape[0] == addition_rewards.shape[0]
        assert addition_rewards.shape[0] == len(episode_step_list)

        final_rewards = RER_ALPHA * raw_rewards + (1 - RER_ALPHA) * addition_rewards
        result_steps = [(episode_info[0], episode_info[1], episode_info[2], episode_info[3], episode_info[4], episode_info[5], episode_info[6], episode_info[7], episode_info[7], final_reward) for episode_info, final_reward in zip(episode_step_list, final_rewards)]
        
        return result_steps
    

    def all_step_size(self):
        return len(self.all_step_list)
            

    def size(self):
        return len(self.red_steps)


class ReplayBuffer:

    # for_all: True为包含红黑双方的回合数据
    def __init__(self, capacity = 20000, data_list = None, for_all = False):
        self.step_list = []   # 只有红方的steps list
        self.all_steps_list = []   # 包含红黑双方的steps list
        self.buffer = collections.deque(maxlen = capacity)  # 队列,先进先出
        if data_list is not None:
            if for_all:
                self.all_steps_list.extend(data_list)
            else:
                self.step_list.extend(data_list)

            for steps in data_list:
                self.buffer.extend(steps)

    def add_round(self, round: Round):
        self.step_list.append(round.red_steps)   # step_list中每个元素是一个round，一个round包含若干steps
        self.buffer.extend(round.red_steps)      # buffer中每个元素是一个独立的step

    def add_round_all(self, round: Round):
        self.all_steps_list.append(round.all_step_list)
        self.buffer.extend(round.all_step_list)

    def round_size(self):  
        return len(self.step_list)
    
    def all_round_size(self):
        return len(self.all_steps_list)
    
    def step_size(self):
        return len(self.buffer)

    def clear(self):
        self.step_list.clear()
        # 不要清self.buffer!!!!!!

    def clear_all_steps(self):
        self.all_steps_list.clear()

    def save(self, expand_data = None):
        if len(self.step_list) == 0:
            return
        
        filedir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'data')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        
        model_version = 0
        if expand_data is not None:
            model_version = expand_data.get('model_version')
        filename = '{}_{}_{}.json'.format(REPLAY_BUFFER_FILE_PREFIX, int(time.time()), model_version)
        filepath = os.path.join(filedir, filename)
        
        with open(filepath, 'w') as f:
            jsonstr = json.dumps(self.step_list)
            f.write(jsonstr)

    def save_all_steps(self, expand_data = None):
        if len(self.all_steps_list) == 0:
            return
        
        filedir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'data_all_steps')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        
        model_version = 0
        if expand_data is not None:
            model_version = expand_data.get('model_version')
        filename = '{}_{}_{}.json'.format(REPLAY_BUFFER_FILE_PREFIX, int(time.time()), model_version)
        filepath = os.path.join(filedir, filename)
        
        with open(filepath, 'w') as f:
            jsonstr = json.dumps(self.all_steps_list)
            f.write(jsonstr)
        

    def sample(self, batch_size):  # 从self.buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        states, pi_list, actions, next_states, raw_rewards, done_list, chapture_list, chapture_rewards, players, re_rewards  = zip(*transitions)
        return states, actions, re_rewards, next_states, done_list, players

    @staticmethod
    def load_from_file(filepath, for_all = False):
        data_list = []
        with open(filepath, 'r') as f:
            jsonstr = f.read()
            data_list = json.loads(jsonstr)
        replay_buffer = ReplayBuffer(data_list = data_list, for_all = for_all)
        return replay_buffer

    @staticmethod
    def load_from_dir(dirpath, for_all = False):
        if not os.path.exists(dirpath):
            return ReplayBuffer(for_all = for_all)
            
        all_data_list = list()
        for filename in os.listdir(dirpath):
            if filename.endswith('json') and filename.startswith(REPLAY_BUFFER_FILE_PREFIX):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    jsonstr = f.read()
                    data_list = json.loads(jsonstr)
                    all_data_list.extend(data_list)

        if len(all_data_list) == 0:
            return ReplayBuffer(for_all = for_all)
        
        replay_buffer = ReplayBuffer(data_list = all_data_list, for_all = for_all)
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