import os
import sys
from copy import deepcopy

# root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_dir)
# # print(f'{sys.path}')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.hoho_config import *
from model.hoho_utils import *
from model.hoho_cchessgame import CChessGame
from model.hoho_mcts import *

class Player:

    def __init__(self):
        self.agent_net = AgentNet()
        self.optimizer = optim.Adam(self.agent_net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
        # self.agent_net.share_memory()

        print(f'{LOG_TAG_AGENT} Player agent created!')
    
    def predict(self, state):
        prob, value = self.agent_net(state)
        return prob, value

    def update(self, states, pis, zs):
        batch_states = states.to(DEVICE)
        batch_pis = torch.tensor(pis, dtype=torch.float).to(DEVICE)
        batch_zs = torch.tensor(pis, dtype=torch.float).to(DEVICE)
        predict_probs, predict_values = self.agent_net(batch_states)
        policy_error = torch.sum(-batch_pis * torch.log(1e-6 + predict_probs), dim=1)
        value_error = (batch_zs - predict_values) ** 2
        loss = (value_error + policy_error).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_models(self):
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filename = os.path.join(dir_path, 'hoho_agent_{}.pth'.format(int(time.time())))
        state = self.agent_net.state_dict()
        torch.save(state, filename)

    def load_models(self, model_path):
        checkpoint = torch.load(model_path)
        self.agent_net.load_state_dict(checkpoint)

    def printModel(self):
        print(f'{LOG_TAG_AGENT} {self.agent_net}')


class AgentNet(nn.Module):

    def __init__(self):
        super(AgentNet, self).__init__()
        self.plane_extractor = PlaneExtractor(IN_PLANES_NUM, FILTER_NUM, RESIDUAL_BLOCK_NUM).to(DEVICE)
        self.value_net = ValueNet(FILTER_NUM, BOARD_POSITION_NUM).to(DEVICE)
        self.policy_net = PolicyNet(FILTER_NUM, BOARD_POSITION_NUM, ACTION_DIM).to(DEVICE)

    def forward(self, state):
        board_features = self.plane_extractor(state)
        win_value = self.value_net(board_features)
        prob = self.policy_net(board_features)
        return prob, win_value


class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, inchannels, outchannels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out)

        return out


class PlaneExtractor(nn.Module):
    """
    棋盘state特征提取，
    """

    def __init__(self, inchannels, outchannels, residual_num):
        super(PlaneExtractor, self).__init__()
        
        self.residual_num = residual_num
        self.conv1 = nn.Conv2d(inchannels, outchannels, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)

        for block in range(self.residual_num):
            setattr(self, 'res{}'.format(block), ResidualBlock(outchannels, outchannels))
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(self.residual_num):
            x = getattr(self, 'res{}'.format(block))(x)

        return x


class PolicyNet(nn.Module):
    """
    输出动作概率分布
    """

    def __init__(self, inplanes, position_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.position_dim = position_dim
        self.conv = nn.Conv2d(inplanes, 2, kernel_size=1) 
        self.bn = nn.BatchNorm2d(2)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(position_dim * 2, action_dim)   # action_dim为动作数，包括pass这个动作

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.position_dim * 2)
        x = self.fc(x)
        action_probs = self.softmax(x)
        return action_probs


class ValueNet(nn.Module):
    """
    输出胜负价值[-1, 1]
    """

    def __init__(self, inplanes, position_dim):
        super(ValueNet, self).__init__()

        self.position_dim = position_dim
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(position_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.position_dim)
        x = F.relu(self.fc1(x))
        win_score = torch.tanh(self.fc2(x))
        return win_score


def self_battle(agent_current, agent_new):

    def red_turn(last_black_action, mcts):
        if last_black_action is not None:
            mcts.take_simulation(agent, game, update_root=False)
            mcts.update_root_with_action(last_black_action)
        pi, action = mcts.take_simulation(agent, game, update_root=True)
        _, _, done = game.step(action)



    accepted = False
    win_count = 0
    for count in range(SELF_BATTLE_NUM):
        done = False
        game = CChessGame()
        red_mcts = MCTS(start_player=PLAYER_RED)
        black_mcts = None
        last_red_action = None
        last_black_action = None
        while not done:
            ##### red turn
            red_pi, red_action = red_mcts.take_simulation(agent_new, game, update_root=(last_black_action is None))
            if last_black_action is None:

                if black_mcts is not None:
                    last_red_action = red_action

                _, _, done = game.step(red_action)
                if done:
                    break
            else:
                red_mcts.update_root_with_action(last_black_action)
                last_black_action = None

            time.sleep(0.1)

            ##### black turn
            if black_mcts is None:
                black_mcts = MCTS(start_player=PLAYER_BLACK, start_state=game.state)
            
            black_pi, black_action = black_mcts.take_simulation(agent_current, game, update_root=(last_red_action is None))
            if last_red_action is None:
                _, _, done = game.step(black_action)
                last_black_action = black_action
                if done:
                    break
            else:
                black_mcts.update_root_with_action(last_red_action)
                last_red_action = None
            
        print(f'{LOG_TAG_AGENT} self battle count={count}')

    return accepted


def train(agent, replay_buffer):
    agent_current = deepcopy(agent)
    agent_new = deepcopy(agent)

    start_time = time.time()
    losses = []
    for epoch in range(EPOCH_NUM):
        batch_states, batch_pis, batch_zs = replay_buffer.sample(BATCH_SIZE)
        planes = [convert_board_to_tensor(state) for state in batch_states]
        planes = torch.stack(planes, dim=0)
        loss = agent_new.update(planes, batch_pis, batch_zs)
        losses.append(loss)

        print(f'{LOG_TAG_AGENT} [train agent] epoch: {epoch} | elapsed: {time.time() - start_time:.3f}s | loss: {loss} | pid: {os.getpid()}')


    accepted = self_battle(agent_current, agent_new)
    if accepted:
        agent_new.save_models()
        # hoho_todo: 通知主进程更新模型


if __name__ == '__main__':
#     # in_channel = 1
#     # out_channel = 1
#     # action_dim = 11

#     # input = torch.ones((1, in_channel, 5, 10)).float()

#     # extractor = PlaneExtractor(in_channel, out_channel)
#     # extractor_output = extractor(input)
#     # print(f'input: {input}')
#     # print(f'extractor output: {extractor_output}')

#     # policy_net = PolicyNet(out_channel, action_dim)
#     # policy_output = policy_net(extractor_output)
#     # print(f'policy_net output: {policy_output}')

#     # value_in_dim = extractor_output.size()[2] * extractor_output.size()[3]
#     # value_net = ValueNet(out_channel, value_in_dim)
#     # value_output = value_net(extractor_output)
#     # print(f'value_new output: {value_output}')

#     # print(f"{os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_models', '12312412')}")

#     state = torch.FloatTensor(torch.ones((BATCH_SIZE, IN_PLANES_NUM, BOARD_WIDTH, BOARD_HEIGHT))).to(DEVICE)
#     player = Player()
#     prob, score = player.predict(state)
#     print(f'prob: {prob.size()}')
#     print(f'score: {score.size()}')

    # myDict = {'red': 1}
    # print(myDict.get('black'))

    p1 = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2]])
    p2 = torch.tensor([[0.8, 0.1, 0.1], [0.5, 0.29, 0.21]])
    print(torch.sum(p1 * torch.log(p2), dim=1))


    v1 = torch.tensor([1, 2], dtype=torch.float)
    v2 = torch.tensor([2, 3])
    print(((v1 - v2) ** 2).view(-1))
