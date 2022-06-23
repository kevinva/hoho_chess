import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 残差网络层数，见hoho_agent.py
RESIDUAL_BLOCK_NUM = 19 # 39

# 特征提取输出通道数，见hoho_agent.py
FILTER_NUM = 128  # 256

# training batch_size
BATCH_SIZE = 16 # 64

# 学习率
LEARNING_RATE = 1e-3

L2_REGULARIZATION = 1e-4

# 计算节点UCB的常数
C_PUCT = 0.2

# EPSILON for Dirichlet noise
NOISE_EPSILON = 0.25

# ALPHA for Dirichlet noise
NOISE_ALPHA = 0.03   

# 进行MCTS的线程数
MCTS_THREAD_NUM = 8

# 每次走子前的模拟次数
MCTS_SIMULATION_NUM = 800 # 1600

# 限制多少步还没分出胜负，则平手
RESTRICT_ROUND_NUM = 100

# Self play次数
SELF_PLAY_NUM = 5000 # 25000

# 计算策略时的temperature系数
POLICY_TEMPERATURE = 1 #1e-3，设置数太小，小心计算inf