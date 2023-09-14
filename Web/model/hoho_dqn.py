import torch
import torch.nn as nn
import torch.nn.functional as F

from hoho_config import *
from hoho_utils import *

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



class BoardPlaneNet(nn.Module):
    """
    棋盘编码网络
    """

    def __init__(self, in_channels, out_channels, residual_num):
        super(BoardPlaneNet, self).__init__()
        
        self.residual_num = residual_num
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        for block in range(self.residual_num):
            setattr(self, 'res{}'.format(block), ResidualBlock(out_channels, out_channels))
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(self.residual_num):
            x = getattr(self, 'res{}'.format(block))(x)

        return x
    


class PolicyNet(nn.Module):
    """
    输出动作概率分布
    """

    def __init__(self, in_planes, position_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.position_dim = position_dim
        self.conv = nn.Conv2d(in_planes, 2, kernel_size = 1) 
        self.bn = nn.BatchNorm2d(2)   # nn.BatchNorm2d是对channel这一维度做归一化处理，输入与输出的shape要保持一致，所以输出channel的维数也为2
        self.fc = nn.Linear(position_dim * 2, action_dim)   # action_dim为动作数，包括pass这个动作
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv(x)
        print(f"1. {x.shape}")
        x = self.bn(x)
        print(f"2. {x.shape}")
        x = F.relu(x)
        print(f"3. {x.shape}")
        x = x.view(-1, self.position_dim * 2)
        print(f"4. {x.shape}")
        x = self.fc(x)
        print(f"5. {x.shape}")
        action_probs = self.softmax(x)
        print(f"6. {x.shape}")

        return action_probs
    

if __name__ == "__main__":
    board_in_channel = IN_PLANES_NUM
    board_out_channel = FILTER_NUM

    board_net = BoardPlaneNet(board_in_channel, board_out_channel, RESIDUAL_BLOCK_NUM)
    action_net = PolicyNet(board_out_channel, BOARD_POSITION_NUM, ACTION_DIM)

    input_tensor = torch.ones((2, board_in_channel, 10, 9))
    board_output = board_net(input_tensor)
    print(f"board_output: {board_output.shape}")

    action_output = action_net(board_output)
    print(f"action_output: {action_output.shape}")


