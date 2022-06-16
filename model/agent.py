import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from gameboard import *

class Player:

    def __init__(self):
        self.plane_extractor = PlaneExtractor(IN_PLANES_NUM, FILTER_NUM, RESIDUAL_BLOCK_NUM).to(DEVICE)
        self.value_net = ValueNet(FILTER_NUM, BOARD_POSITION_NUM).to(DEVICE)
        self.policy_net = PolicyNet(FILTER_NUM, BOARD_POSITION_NUM, ACTION_DIM).to(DEVICE)
        self.passed = False
    
    def predict(self, state):
        board_features = self.plane_extractor(state)
        win_value = self.value_net(board_features)
        prob = self.policy_net(board_features)
        return prob, win_value

    def printModel(self):
        print('extractor: ', self.plane_extractor)
        print('value head: ', self.value_net)
        print('policy head: ', self.policy_net)

    def save_models(self, state, current_time):
        for model in ['plane_extractor', 'policy_net', 'value_net']:  # 注意跟属性名对应
            self._save_checkpoint(getattr(self, model), model, state, current_time)

    def _save_checkpoint(self, model, filename, state, current_time):
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_models', current_time)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filename = os.path.join(dir_path, '{}-{}.pth.tar'.format(filename, state['version']))
        state['model'] = model.state_dict()
        torch.save(state, filename)

    def load_models(self, path, models):
        names = ['plane_extractor', 'policy_net', 'value_net']
        for i in range(0, len(models)):
            checkpoint = torch.load(os.path.join(path, models[i]))
            model = getattr(self, names[i])
            model.load_state_dict(checkpoint['model'])
            return checkpoint  # hoho: 确定就在循环内return?


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
        out += residual
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

if __name__ == '__main__':
    # in_channel = 1
    # out_channel = 1
    # action_dim = 11

    # input = torch.ones((1, in_channel, 5, 10)).float()

    # extractor = PlaneExtractor(in_channel, out_channel)
    # extractor_output = extractor(input)
    # print(f'input: {input}')
    # print(f'extractor output: {extractor_output}')

    # policy_net = PolicyNet(out_channel, action_dim)
    # policy_output = policy_net(extractor_output)
    # print(f'policy_net output: {policy_output}')

    # value_in_dim = extractor_output.size()[2] * extractor_output.size()[3]
    # value_net = ValueNet(out_channel, value_in_dim)
    # value_output = value_net(extractor_output)
    # print(f'value_new output: {value_output}')

    # print(f"{os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_models', '12312412')}")

    state = torch.FloatTensor(torch.ones((BATCH_SIZE, IN_PLANES_NUM, BOARD_WIDTH, BOARD_HEIGHT))).to(DEVICE)
    player = Player()
    prob, score = player.predict(state)
    print(f'prob: {prob.size()}')
    print(f'score: {score.size()}')