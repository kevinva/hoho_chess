
if __name__ == '__main__':
	import os
	from . import chess, alpha_beta
	board_key = []
	with open('board.txt') as fp:
		for line in fp:
			if line!='':
				player, ctype, x, y = line.split(' ')
				board_key.append([player, ctype, int(x), int(y)])
	os.unlink('board.txt')
	board = chess.board_from_key(board_key)
	move = alpha_beta.auto_move(board)
	with open('move.txt', 'w') as fp:
		fp.write(' '.join([str(x) for x in move]))
