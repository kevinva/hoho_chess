import torch


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESIDUAL_BLOCK_NUM = 256