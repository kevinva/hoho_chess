import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .hoho_config import *
from .hoho_utils import *



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

        self.board_layer = BoardPlaneLayer(in_channels = IN_PLANES_NUM, out_channels = FILTER_NUM, residual_num = RESIDUAL_BLOCK_NUM)
        self.action_layer = ActionLayer(in_planes = FILTER_NUM, position_dim = BOARD_POSITION_NUM, action_dim = ACTION_DIM)

    def forward(self, state):
        board_features = self.board_layer(state)
        q_value = self.action_layer(board_features)
        ### 最终输出动作的Q值

        return q_value


class DQN:
    ''' DQN算法 '''

    def __init__(self, action_dim, learning_rate, gamma, epsilon, target_update, device):
        
        self.version = 0
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # 原始Q网络
        self.q_net = QNetwork().to(device)  

        # 目标Q网络
        self.target_q_net = QNetwork().to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.target_update = target_update
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state_str):  # epsilon-贪婪策略采取动作
        all_legal_actions = get_legal_actions(state_str, PLAYER_RED)
        pis = np.zeros((ACTION_DIM,))

        if np.random.random() < self.epsilon:
            print("random action!")

            legal_action_dim = len(all_legal_actions)
            action_idx = np.random.randint(legal_action_dim)
            action = all_legal_actions[action_idx]
            action_idx = ACTIONS_2_INDEX[action]
            pis[action_idx] = 1.0
        else:
            print("argmax action!")
            
            state_tensor = convert_board_to_tensor(state_str).unsqueeze(0).to(self.device)
            action_values = self.q_net(state_tensor)
            action_values = action_values.to(torch.device('cpu')).detach().numpy()[0]

            q_values = np.zeros((ACTION_DIM,))
            for action in all_legal_actions:
                action_idx = ACTIONS_2_INDEX[action]
                q_values[action_idx] = action_values[action_idx]
            action_idx = q_values.argmax()
            action = INDEXS_2_ACTION[action_idx]
            pis = q_values / np.sum(q_values)

        return action, pis
    
    def update(self, transition_dict):
        planes = [convert_board_to_tensor(state) for state in transition_dict['states']]
        states_tensor = torch.stack(planes, dim = 0).to(self.device)
        
        actions_index_list = []
        for action in transition_dict['actions']:
            action_idx = ACTIONS_2_INDEX[action]
            actions_index_list.append(action_idx)
        actions_index_tensor = torch.tensor(actions_index_list).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device)

        next_planes = [convert_board_to_tensor(state) for state in transition_dict['next_states']]
        next_states_tensor = torch.stack(next_planes, dim = 0).to(self.device)

        dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1, 1).to(self.device)

        # 为了方便，直接用action的索引作为输入，暂没有编码action
        q_values = self.q_net(states_tensor).gather(1, actions_index_tensor)  # Q值 (gather用法参考：https://blog.csdn.net/qq_38964360/article/details/131550919)

        # 下个状态的最大Q值
        target_action_values_tensor = self.target_q_net(next_states_tensor)
        batch_size = target_action_values_tensor.shape[0]
        next_states_str_list = transition_dict['next_states']
        next_legal_actions_list = [get_legal_actions(next_state, PLAYER_RED) for next_state in next_states_str_list]
        target_action_values_formatted = torch.zeros((batch_size, ACTION_DIM))
        for i in range(batch_size):
            legal_actions = next_legal_actions_list[i]
            for action in legal_actions:
                action_idx = ACTIONS_2_INDEX[action]
                target_action_values_formatted[i][action_idx] = target_action_values_tensor[i][action_idx]
        max_next_q_values = target_action_values_formatted.max(1)[0].view(-1, 1)
        
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # print(f'loss: {dqn_loss.item()}')
            LOGGER.info(f'Update Target network!')
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        return dqn_loss.item()


    def save_model(self):
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filepath = os.path.join(dir_path, '{}_{}_{}.pth'.format(MODEL_FILE_PREFIX, int(time.time()), self.version))
        state = self.q_net.state_dict()
        torch.save(state, filepath)

        return filepath

    def load_model_from_path(self, model_path):
        filename = os.path.basename(model_path)
        if not filename.startswith(MODEL_FILE_PREFIX):
            return

        filename = filename.split('.')[0]
        items = filename.split('_')
        if len(items) == 3:
            self.version = int(items[2])

        checkpoint = torch.load(model_path)
        self.q_net.load_state_dict(checkpoint)
        self.target_q_net.load_state_dict(checkpoint)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.learning_rate)
    
    def update_version(self):
        self.version += 1

    def printModel(self):
        LOGGER.info(f'{self.q_net}')


    def set_train_mode(self):
        self.q_net.train()
    
    def set_eval_mode(self):
        self.q_net.eval()


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


    ########################################################

    # # 创建一个输入张量
    # input = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # # 创建一个索引张量，指定要收集的元素的位置
    # index = torch.tensor([[0, 1], [1, 0], [2, 1]])
    # index_column = torch.tensor([[0, 1], [1, 0]])
    # index_column2 = torch.tensor([[0], [1]])

    # # 在维度0上使用gather函数
    # # result = torch.gather(input, 0, index)
    # result = torch.gather(input, 1, index_column2)
    # print(result)

    ########################################################

    gamma = 0.99
    lr = 5e-5
    epsilon = 0.1

    dqn = DQN(ACTION_DIM, lr, gamma, epsilon, 100, DEVICE)
    state = INIT_BOARD_STATE
    for i in range(100):
        action = dqn.take_action(state)
        print(f"    action: {action}")



