
import sys, time

def auto_move(board):
	assert len(sys.argv)==2, ('程序需要有一个参数',sys.argv)
	if sys.argv[1]=='min_max':
		from . import min_max
		return min_max.auto_move(board)
	elif sys.argv[1]=='alpha_beta':
		from . import alpha_beta
		# print(f'argv: {sys.argv}')
		return alpha_beta.auto_move(board)
	else:
		print(f'正在调用 {sys.argv[1]}')
		return auto_move_cpp(board, sys.argv[1])

def auto_move_cpp(board, cmd):
	with open('board.txt', 'w') as fp:
		for _, c in board.board_map.items():
			fp.write(f'{c.player} {c.type} {c.x} {c.y}\n')
	import os
	os.system(cmd)	# 读取并删除 board.txt 产生 move.txt
	while not os.path.isfile('move.txt'):
		time.sleep(.1)
	with open('move.txt') as fp:
		move = fp.read()
	os.unlink('move.txt')
	move = move.split(' ')
	return [int(x) for x in move]
	