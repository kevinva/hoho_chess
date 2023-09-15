import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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



class BoardPlaneLayer(nn.Module):
    """
    棋盘编码网络
    """

    def __init__(self, in_channels, out_channels, residual_num):
        super(BoardPlaneLayer, self).__init__()
        
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
    


class ActionLayer(nn.Module):
    """
    输出动作Q值预测
    """

    def __init__(self, in_planes, position_dim, action_dim):
        super(ActionLayer, self).__init__()

        self.position_dim = position_dim
        self.conv = nn.Conv2d(in_planes, 2, kernel_size = 1) 
        self.bn = nn.BatchNorm2d(2)   # nn.BatchNorm2d是对channel这一维度做归一化处理，输入与输出的shape要保持一致，所以输出channel的维数也为2
        self.fc = nn.Linear(position_dim * 2, action_dim)   # action_dim为动作数，包括pass这个动作
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.position_dim * 2)
        result = self.fc(x)

        return result
    

class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()

        self.plane_net = BoardPlaneLayer(in_channels = IN_PLANES_NUM, out_channels = FILTER_NUM, residual_num = RESIDUAL_BLOCK_NUM)
        self.policy_net = ActionLayer(in_planes = FILTER_NUM, position_dim = BOARD_POSITION_NUM, action_dim = ACTION_DIM)

    def forward(self, state):
        board_features = self.plane_net(state)
        prob = self.policy_net(board_features)

        return prob


class DQN:
    ''' DQN算法 '''
    def __init__(self, action_dim, learning_rate, gamma, epsilon, target_update, device):

        self.action_dim = action_dim

        # 原始Q网络
        self.q_net = QNetwork().to(device)  

        # 目标Q网络
        self.target_q_net = QNetwork().to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标Q网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype = torch.float).to(self.device)
            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype = torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype = torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # print(f'loss: {dqn_loss.item()}')
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    

if __name__ == "__main__":
    # board_in_channel = IN_PLANES_NUM
    # board_out_channel = FILTER_NUM

    # board_net = BoardPlaneLayer(board_in_channel, board_out_channel, RESIDUAL_BLOCK_NUM)
    # action_net = ActionLayer(board_out_channel, BOARD_POSITION_NUM, ACTION_DIM)

    # input_tensor = torch.ones((2, board_in_channel, 10, 9))
    # board_output = board_net(input_tensor)
    # print(f"board_output: {board_output.shape}")

    # action_output = action_net(board_output)
    # print(f"action_output: {action_output.shape}")


    # 创建一个输入张量
    input = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # 创建一个索引张量，指定要收集的元素的位置
    index = torch.tensor([[0, 1], [1, 0], [2, 1]])
    index_column = torch.tensor([[0, 1], [1, 0]])

    # 在维度0上使用gather函数
    # result = torch.gather(input, 0, index)
    result = torch.gather(input, 1, index_column)
    print(result)






