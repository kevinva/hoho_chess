from hoho_utils import *
from hoho_config import *

class CChessGame:

    def __init__(self):
        self.reset()

    def step(self, action):
        pass

    def reset(self):
        self.state = INIT_BOARD_STATE
        self.restrict_count = RESTRICT_ROUND_NUM
        self.winner = None
        self.current_player = PLAYER_RED


