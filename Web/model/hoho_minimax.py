from copy import deepcopy
from hoho_utils import *
from hoho_cchessgame import *


def minimax(game, depth_count):
    # 若预先估计后3步，则depth_count = 3 - 1 = 2， 以此类推

    def max_value(game, done, depth):
        if depth == 0 or done:
            score = game.debug_evaluate()

            print(f"depth: {depth_count - depth}, max_score: {score}, best_move: {None}")

            return score, None   # hoho_todo

        max_score = -float('inf')
        best_move = None
        for move in get_legal_actions(game.state, PLAYER_RED):
            game_copy = deepcopy(game)
            state_new, final_reward, done_new = game_copy.step(move)
            score, _ = min_value(game_copy, done_new, depth - 1)
            if score > max_score:
                max_score = score
                best_move = move

        print(f"depth: {depth_count - depth}, max_score: {max_score}, best_move: {best_move}")

        return max_score, best_move


    def min_value(game, done, depth):
        if depth == 0 or done:
            score = game.debug_evaluate()

            print(f"depth: {depth_count - depth}, min_score: {score}, best_move: {None}")

            return score, None    # hoho_todo

        min_score = float('inf')
        best_move = None
        for move in get_legal_actions(game.state, PLAYER_RED):
            game_copy = deepcopy(game)
            state_new, final_reward, done_new = game_copy.step(move)
            score, _ = max_value(game_copy, done_new, depth - 1)
            if score < min_score:
                min_score = score
                best_move = move

        print(f"depth: {depth_count - depth}, min_score: {min_score}, best_move: {best_move}")

        return min_score, best_move

    if game.current_player == PLAYER_RED:
        score, best_move = max_value(game, False, depth_count)
        LOGGER.info(f"score = {score}, best_move = {best_move}")
        return score, best_move
    else:
        score, best_move = min_value(game, False, depth_count)
        return score, best_move
    

if __name__ == "__main__":
    game = CChessGame()
    minimax(game, 2)

