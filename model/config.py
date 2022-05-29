import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GOBAN_SIZE = 9
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
HISTORY_NUM = 7  # 前多少步历史走棋状态
IN_PLANES_NUM = (HISTORY_NUM + 1) * 2 + 1  # 正反两方，所以乘以2， 括号外+1为当前哪方下棋的状态
ACTION_POSITION_NUM = BOARD_WIDTH * BOARD_HEIGHT
ACTION_DIM = 2086  # 用create_all_moves方法得出走法数量
RESIDUAL_BLOCK_NUM = 19 # 39
FILTER_SIZE = 128  # 256
BATCH_SIZE = 64

C_PUCT = 0.2
P_EPSILON = 0.25
MCTS_PARALLEL = 4  # 线程数
MCTS_SIMULATION = 64  # 模拟次数
BATCH_SIZE_EVAL = 2