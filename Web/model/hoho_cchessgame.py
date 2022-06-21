from hoho_utils import *
from hoho_config import *

class CChessGame:

    def __init__(self):
        self.reset()

    def step(self, action):
        state_new = do_action_on_board(self.state, action)
        value = 0
        done = False
        if 'K' not in state_new:
            value = -1
            done = True
            self.winner = PLAYER_BLACK
        elif 'k' not in state_new:
            value = 1
            done = True
            self.winner = PLAYER_RED

        return state_new, value, done

    def reset(self):
        self.state = INIT_BOARD_STATE
        self.restrict_count = RESTRICT_ROUND_NUM
        self.winner = None
        self.current_player = PLAYER_RED




if __name__ == '__main__':
    s = 'ERIOC<VGK1234q24ds'
    print(('k' in s))