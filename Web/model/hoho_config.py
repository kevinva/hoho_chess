import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import torch

# multiprocess + GPU，loss没降
# multiprocess + CPU，loss有降
# no multiprocess + GPU，loss有降
# no multiprocess + 
# 难道是因为GPU为单卡的原因？还是用多线程吧
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# DEVICE = torch.device('cpu')

# 残差网络层数，见hoho_agent.py
RESIDUAL_BLOCK_NUM = 40

# 特征提取输出通道数，见hoho_agent.py
FILTER_NUM = 256

# training batch_size
BATCH_SIZE = 64

# 学习率
LEARNING_RATE = 5e-5

# Loss用的L2范数
L2_REGULARIZATION = 1e-4

# 训练的epoch数
EPOCH_NUM = 100

# 计算节点UCB的常数
C_PUCT = 0.2

# EPSILON for Dirichlet noise
NOISE_EPSILON = 0.25

# ALPHA for Dirichlet noise
NOISE_ALPHA = 0.03   

# 进行MCTS的线程数
MCTS_THREAD_NUM = 8

# 每次走子前的模拟次数
MCTS_SIMULATION_NUM = 800 #1600

# 计算策略时的temperature系数
POLICY_TEMPERATURE = 1 #1e-3，设置数太小，小心计算inf

# 限制多少步还没分出胜负，则平手
RESTRICT_ROUND_NUM = 100

# 自博弈时的限制步数
SELF_BATTLE_RESTRICT_ROUND_NUM = 100

# 网络训练完成后自博弈的次数
SELF_BATTLE_NUM = 200 # 400

# 训练的网络与当前网络的自博弈，超过该阈值即替换当前网络
SELF_BATTLE_WIN_RATE = 0.55

# 控制某些日志是否打印
DEBUG = True


# DQN用
GAMMA = 0.98   # 折扣因子
EPSILON_G = 0.2 # epsilon贪婪策略
TARGET_UPDATE_COUNT = 10  # 更新目标网络的频率

# 奖励重分配参数
RER_LAMBDA = 0.90
RER_WINDOW_SIZE = 2  # 前后window_size个位置都要考虑


LOG_TAG_AGENT = '[hoho_agent]'
LOG_TAG_CCHESSGAME = '[hoho_cchessgame]'
LOG_TAG_REPLAY_BUFFER = '[hoho_replay_buffer]'
LOG_TAG_MCTS = '[hoho_mcts]'
LOG_TAG_SERV = '[hoho_serv]'

KEY_MSG_ID = 'msg_id'
KEY_AGENT_ACCEPT = 'agent_accept'
KEY_MODEL_PATH = 'model_path'
AGENT_MSG_ID_TRAIN_FINISH = 1
AGENT_MSG_ID_SELF_BATTLE_FINISH = 2
REPLAY_BUFFER_FILE_PREFIX = 'D3QNAB'
MODEL_FILE_PREFIX = 'D3QN'