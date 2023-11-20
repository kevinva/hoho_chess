from copy import deepcopy
from model.hoho_utils import *


def minimax(game, done, depth):

    def max_value(game, done, depth):
        if depth == 0 or done:
            return game.debug_evaluate(), None   # hoho_todo

        max_score = -float('inf')
        best_move = None
        for move in get_legal_actions(game.state, PLAYER_RED):
            game_copy = deepcopy(game)
            state_new, final_reward, done_new = game_copy.step(move)
            score, _ = min_value(game_copy, done_new, depth - 1)
            if score > max_score:
                max_score = score
                best_move = move

        print(f"depth: {depth}, max_score: {max_score}, best_move: {best_move}")

        return max_score, best_move


    def min_value(game, done, depth):
        if depth == 0 or done:
            return game.debug_evaluate(), None    # hoho_todo

        min_score = float('inf')
        best_move = None
        for move in get_legal_actions(game.state, PLAYER_RED):
            game_copy = deepcopy(game)
            state_new, final_reward, done_new = game_copy.step(move)
            score, _ = max_value(game_copy, done_new, depth - 1)
            if score < min_score:
                min_score = score
                best_move = move

        print(f"depth: {depth}, min_score: {min_score}, best_move: {best_move}")

        return min_score, best_move

    if game.current_player == PLAYER_RED:
        return max_value(game, done, depth)
    else:
        return min_value(game, done, depth)