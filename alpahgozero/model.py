import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RESIDUAL_BLOCK_NUM

class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)

        return out


class Extractor(nn.Module):
    """
    棋盘state特征提取，
    """

    def __init__(self, inplanes, outplanes):
        super(Extractor, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        for block in range(RESIDUAL_BLOCK_NUM):
            setattr(self, 'res{}'.format(block), ResidualBlock(outplanes, outplanes))
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(RESIDUAL_BLOCK_NUM):
            x = getattr(self, 'res{}'.format(block))(x)

        return x


class PolicyNet(nn.Module):
    """
    输出动作概率分布
    """

    def __init__(self, inplanes, action_dim):
        super(PolicyNet, self).__init__()

        self.action_dim = action_dim
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)  # hoho: 原论文不是2个filter么？
        self.bn = nn.BatchNorm2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(action_dim - 1, action_dim)   # action_dim为动作数，包括pass这个动作

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.action_dim - 1)
        x = self.fc(x)
        action_probs = self.softmax(x)
        return action_probs


class ValueNet(nn.Module):
    """
    输出胜负价值[-1, 1]
    """

    def __init__(self, inplanes, outplanes):
        super(ValueNet, self).__init__()

        self.outplanes = outplanes
        self.cov = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(outplanes - 1, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = F.relu(self.fc1(x))
        winning = F.tanh(self.fc2(x))
        return winning

if __name__ == '__main__':
    input = torch.ones((1, 1, 5, 10)).float()
    extractor = Extractor(1, 1)
    output = extractor(input)
    print(f'input: {input}')
    print(f'extractor output: {output}')

    policy_net = PolicyNet(1, 11)
    output = policy_net(output)
    print(f'policy_net output: {output}')
