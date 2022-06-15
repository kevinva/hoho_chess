import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GOBAN_SIZE = 9

HISTORY_NUM = 7  # 前多少步历史走棋状态
IN_PLANES_NUM = (HISTORY_NUM + 1) * 2 + 1  # 正反两方，所以乘以2， 括号外+1为当前哪方下棋的状态
RESIDUAL_BLOCK_NUM = 19 # 39
FILTER_NUM = 128  # 256
BATCH_SIZE = 16 # 64
SIMULATION_NUM = 1600

C_PUCT = 0.2
NOISE_EPSILON = 0.25  # EPSILON for Dirichlet noise
NOISE_ALPHA = 0.03    # ALPHA for Dirichlet noise
VIRTUAL_LOSS = 3
MCTS_THREAD_NUM = 4  # 线程数
MCTS_SIMULATION_NUM = 800  # 模拟次数
RESTRICT_ROUND_NUM = 100   # 限制多少步还没分出胜负，则平手