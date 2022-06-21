import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 残差网络层数，见hoho_agent.py
RESIDUAL_BLOCK_NUM = 19 # 39

# 特征提取输出通道数，见hoho_agent.py
FILTER_NUM = 128  # 256

# batch_size
BATCH_SIZE = 16 # 64

# 计算节点UCT的常数
C_PUCT = 0.2

# EPSILON for Dirichlet noise
NOISE_EPSILON = 0.25

# ALPHA for Dirichlet noise
NOISE_ALPHA = 0.03   

# 进行MCTS的虚拟损失，backup前记得恢复过来
VIRTUAL_LOSS = 3

# 进行MCTS的线程数
MCTS_THREAD_NUM = 4

# 每次走子前的模拟次数
MCTS_SIMULATION_NUM = 800  

# 限制多少步还没分出胜负，则平手
RESTRICT_ROUND_NUM = 100   