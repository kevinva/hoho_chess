import numpy as np
import random
from config import*

def sample_rotation(state, num=8):
    """ Apply a certain number of random transformation to the input state """

    ## Create the dihedral group of a square with all the operations needed
    ## in order to get the specific transformation and randomize their order
    dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
                ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
                (np.flipud,  (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]
    random.shuffle(dh_group)

    states = []
    boards = (HISTORY_NUM + 1) * 2 ## Number of planes to rotate

    for idx in range(num):
        new_state = np.zeros((boards + 1, GOBAN_SIZE, GOBAN_SIZE,))
        new_state[:boards] = state[:boards]

        ## Apply the transformations in the tuple defining how to get
        ## the desired dihedral rotation / transformation
        for grp in dh_group[idx]:
            for i in range(boards):
                if isinstance(grp, tuple):
                    new_state[i] = grp[0](new_state[i], k=grp[1])
                elif grp is not None:
                    new_state[i] = grp(new_state[i])

        new_state[boards] = state[boards]
        states.append(new_state)
    
    if len(states) == 1:
        return np.array(states[0])
    return np.array(states)


# 所有合法走子，一共有2086个走法
def create_all_moves():
    moves = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 士的走法
    mandarins_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                        'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    
    # 象的走法
    elephants_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                        'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                        'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                        'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    cols = len(letters)
    rows = len(numbers)
    for c in range(cols):
        for r in range(rows):
            # 这里假设起始点为（c, r）,以下计算能走的目标点
            destinations = [(t, r) for t in range(cols)] + \
                           [(c, t) for t in range(rows)] + \
                           [(c + a, r + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]] # 马走日

        
            for (c2, r2) in destinations:
                if (c, r) != (c2, r2) and c2 in range(cols) and r2 in range(rows):
                    move = letters[c] + numbers[r] + letters[c2] + numbers[r2]
                    moves.append(move)
            
    moves.extend(mandarins_labels)
    moves.extend(elephants_labels)

    return moves



if __name__ == '__main__':
    moves = create_all_moves()
    print(len(moves))
    print(moves[:20])
