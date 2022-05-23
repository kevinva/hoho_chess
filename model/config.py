import torch


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

GOBAN_SIZE = 9
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
HISTORY_NUM = 7  # 前多少步历史走棋状态
IN_PLANES_NUM = (HISTORY_NUM + 1) * 2 + 1  # 正反两方，所以乘以2， 括号外+1为当前哪方下棋的状态
OUT_PLANES_NUM = 10
ACTION_DIM = (GOBAN_SIZE ** 2) + 1  # hoho: 需要一个函数算出所有落子动作空间/这个ACTION_DIM可能与OUT_PLANES_NUM有所相关
RESIDUAL_BLOCK_NUM = 256

C_PUCT = 0.2
P_EPSILON = 0.25
MCTS_PARALLEL = 4  # 线程数
MCTS_SIMULATION = 64  # 模拟次数
BATCH_SIZE_EVAL = 2