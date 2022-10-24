import os
import sys
import random
import threading
from copy import deepcopy


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
# # print(f'{sys.path}')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.hoho_config import *
from model.hoho_utils import *
from model.hoho_cchessgame import *
from model.hoho_mcts import *

class Player:

    def __init__(self):
        self.agent_net = AgentNet().to(DEVICE)
        self.optimizer = optim.Adam(self.agent_net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
        # self.agent_net.share_memory()
        self.version = 0

        LOGGER.info('Player agent created!')

    def set_train_mode(self):
        self.agent_net.train()
    
    def set_eval_mode(self):
        self.agent_net.eval()
    
    def predict(self, state):
        prob, value = self.agent_net(state)
        return prob, value

    def update(self, states, pis, zs):
        batch_states = states.to(DEVICE)
        batch_pis = pis.to(DEVICE)
        batch_zs = zs.view(-1, 1).to(DEVICE)
        predict_probs, predict_values = self.agent_net(batch_states)
        policy_error = torch.sum(-batch_pis * torch.log(predict_probs.clamp(min=1e-6)), dim=1)   # clamp(min=1e-6)防止log(0)
        value_error = (batch_zs - predict_values) ** 2
        loss = (value_error + policy_error).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self):
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output', 'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filepath = os.path.join(dir_path, '{}_{}_{}.pth'.format(MODEL_FILE_PREFIX, int(time.time()), self.version))
        state = self.agent_net.state_dict()
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
        self.agent_net.load_state_dict(checkpoint)
        self.optimizer = optim.Adam(self.agent_net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    
    def update_version(self):
        self.version += 1

    def printModel(self):
        LOGGER.info(f'{self.agent_net}')


class AgentNet(nn.Module):

    def __init__(self):
        super(AgentNet, self).__init__()
        self.plane_extractor = PlaneExtractor(IN_PLANES_NUM, FILTER_NUM, RESIDUAL_BLOCK_NUM)
        self.value_net = ValueNet(FILTER_NUM, BOARD_POSITION_NUM)
        self.policy_net = PolicyNet(FILTER_NUM, BOARD_POSITION_NUM, ACTION_DIM)

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


def self_battle(agent_current, agent_new, use_mcts=True, msg_queue=None):
    """
    新训练网络与当前网络自博弈
    
    Arguments:
        use_mcts: 是否使用MCTS进行策略生成。False则直接使用神经网络预测的走子策略。
    """

    LOGGER.info('start self battle!!!')

    def red_turn(last_black_action, mcts, agent, game, use_mcts=True):
        done = False
        red_action = None
        if use_mcts:
            if last_black_action is not None:
                if not mcts.is_current_root_expanded():
                    mcts.take_simulation(agent, game, update_root=False)
                mcts.update_root_with_action(last_black_action)
            pi, red_action = mcts.take_simulation(agent, game, update_root=True)
            _, _, done = game.step(red_action)
        else:
            planes = convert_board_to_tensor(game.state).unsqueeze(0).to(DEVICE)
            pis, _ = agent.predict(planes)
            pi = pis.to(torch.device('cpu')).detach().numpy()[0]

            all_legal_actions = get_legal_actions(game.state, PLAYER_RED)
            probas = np.zeros((ACTION_DIM,))
            for action in all_legal_actions:
                action_idx = ACTIONS_2_INDEX[action]
                probas[action_idx] = pi[action_idx]
            probas = probas / np.sum(probas) 

            action_idx = np.random.choice(ACTION_DIM, p=probas)
            red_action = INDEXS_2_ACTION[action_idx]
            _, _, done = game.step(red_action)

        return red_action, done

    def black_turn(last_red_action, mcts, agent, game, expanded, use_mcts=True):
        done = False
        black_action = None
        if use_mcts:
            if (last_red_action is not None) and expanded:
                if not mcts.is_current_root_expanded():
                    mcts.take_simulation(agent, game, update_root=False)
                mcts.update_root_with_action(last_red_action)
            pi, black_action = mcts.take_simulation(agent, game, update_root=True)
            _, _, done = game.step(black_action)
        else:
            state = flip_board(game.state)
            planes = convert_board_to_tensor(state).unsqueeze(0).to(DEVICE)
            pis, _ = agent.predict(planes)
            pi = pis.to(torch.device('cpu')).detach().numpy()[0]
            pi = flip_action_probas(pi)

            all_legal_actions = get_legal_actions(game.state, PLAYER_BLACK)
            probas = np.zeros((ACTION_DIM,))
            for action in all_legal_actions:
                action_idx = ACTIONS_2_INDEX[action]
                probas[action_idx] = pi[action_idx]
            probas = probas / np.sum(probas)
            
            action_idx = np.random.choice(ACTION_DIM, p=probas)
            black_action = INDEXS_2_ACTION[action_idx]
            _, _, done = game.step(black_action)

        return black_action, done

    accepted = False
    win_count = 0
    draw_count = 0
    loss_count = 0
    for match_count in range(SELF_BATTLE_NUM):
        start_time = time.time()

        done = False
        game = CChessGame()
        red_mcts = MCTS(start_player=PLAYER_RED)
        black_mcts = None
        last_red_action = None
        last_black_action = None
        black_expanded = False
        round_count = 0
        while not done:
            last_red_action, done = red_turn(last_black_action, red_mcts, agent_new, game, use_mcts)
            LOGGER.info(f'[tid={threading.currentThread().ident}] Self battle! rounds: {round_count + 1} / matches: {match_count + 1} | red turn: action={last_red_action}, state={game.state}')

            if done:
                break

            time.sleep(random.random() * 2)  # 随机让线程睡眠几秒，让其他线程有机会执行

            if black_mcts is None:
                black_mcts = MCTS(start_player=PLAYER_BLACK, start_state=game.state)
            last_black_action, done = black_turn(last_red_action, black_mcts, agent_current, game, black_expanded, use_mcts)
            LOGGER.info(f'[tid={threading.currentThread().ident}] Self battle! rounds: {round_count + 1} / matches: {match_count + 1} | black turn: action={last_black_action}, state={game.state}')
           
            if done:
                break

            if not black_expanded:
                black_expanded = True

            round_count += 1
            if round_count > SELF_BATTLE_RESTRICT_ROUND_NUM:   # 超过步数，提前结束
                done = True
                break
       
        if done:
            if game.winner == PLAYER_RED:
                win_count += 1
            elif game.winner == PLAYER_BLACK:
                loss_count += 1
            else:
                draw_count += 1

        win_rate_included_draw = 0
        win_rate_not_included_draw = 0
        if (match_count + 1) > 0:
            win_rate_included_draw = win_count / (match_count + 1)
        if (win_count + loss_count) > 0:
            win_rate_not_included_draw = win_count / (win_count + loss_count)
        LOGGER.info(f'[tid={threading.currentThread().ident}] Self battle! win count: {win_count} | match count: {match_count + 1} | win rate (draw included) = {win_rate_included_draw} | win rate (draw not included) = {win_rate_not_included_draw} | win = {win_count}, loss = {loss_count}, draw = {draw_count} | elapse: {time.time() - start_time:.3f}s')
    
    accepted = ((win_count / SELF_BATTLE_NUM) >= SELF_BATTLE_WIN_RATE)
    model_path = None
    LOGGER.info(f'self battle result: accepted? {accepted}')
 
    if accepted:
        agent_new.update_version()
        model_path = agent_new.save_model()
        # hoho_todo: 通知主线程更新模型

    msg_queue.put({KEY_MSG_ID: AGENT_MSG_ID_SELF_BATTLE_FINISH, KEY_AGENT_ACCEPT: accepted, KEY_MODEL_PATH: model_path})

    return accepted


def train(agent):
    root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir_path = os.path.join(root_dir_path, 'output', 'data')
    if not os.path.exists(data_dir_path):
        return
    
    if len(os.listdir(data_dir_path)) < 5:
        return

    train_dataset = ChessDataset.load_from_dir(data_dir_path, version=agent.version)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_batch_len = len(train_dataloader)
    
    LOGGER.info(f'train batch len = {train_batch_len}, total train data size = {len(train_dataset)}, version = {agent.version}')

    # agent_current = deepcopy(agent)
    # agent_new = deepcopy(agent)

    agent_current = Player()
    agent_current.version = agent.version
    agent_current.agent_net.load_state_dict(agent.agent_net.state_dict())
    agent_new = Player()
    agent_new.version = agent.version
    agent_new.agent_net.load_state_dict(agent.agent_net.state_dict())

    agent_new.set_train_mode()
    for epoch in range(EPOCH_NUM):
        start_time = time.time()
        train_loss = 0.0
        for batch_i, (batch_states, batch_pis, batch_zs) in enumerate(train_dataloader):
            planes = [convert_board_to_tensor(state) for state in batch_states]
            batch_planes = torch.stack(planes, dim=0)
            loss = agent_new.update(batch_planes, batch_pis, batch_zs)
            train_loss += loss

        LOGGER.info(f'[tid={threading.currentThread().ident}] Training! epoch: {epoch + 1} | elapse: {(time.time() - start_time):.3f}s | loss: {(train_loss / train_batch_len):.6f}')

    agent_new.set_eval_mode()

    return agent_new, agent_current


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

    # p1 = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2]])
    # p2 = torch.tensor([[0.8, 0.1, 0.1], [0.5, 0.29, 0.21]])
    # print(torch.sum(p1 * torch.log(p2), dim=1))


    # v1 = torch.tensor([1, 2], dtype=torch.float)
    # v2 = torch.tensor([2, 3])
    # print(((v1 - v2) ** 2).view(-1))

    # dirpath = '../output/data/'
    # rb = ReplayBuffer.load_from_dir(dirpath)
    # print(sum(list(rb.buffer)[100][1]))

    # testt = torch.tensor([1, 2, 3, 4, 5, -1, 0.1, -2], dtype=torch.float)
    # print(testt.clamp(min=1e-3))
    # print(testt.unsqueeze(0))

    from thop import profile
    from thop import clever_format

    model = AgentNet()
    input = torch.rand((1, IN_PLANES_NUM, BOARD_HEIGHT, BOARD_WIDTH))
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'flops: {flops}, params: {params}')

