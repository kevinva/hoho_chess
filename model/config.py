import torch


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BOARD_WIDTH = 9
BOARD_HEIGHT = 10
HISTORY_NUM = 7
IN_PLANES_NUM = (HISTORY_NUM + 1) * 2
OUT_PLANES_NUM = 10
ACTION_DIM = 0  # hoho: 需要一个函数算出所有落子动作数量
RESIDUAL_BLOCK_NUM = 256