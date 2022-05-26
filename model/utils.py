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



# if __name__ == '__main__':
    # moves = create_all_moves()
    # print(len(moves))
    # print(moves[:20])
