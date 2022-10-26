import json
import time
from . import chess
from . import spinner
from . import ajax
import heapq

HOHO_RESTRICT_ROUND_NUM = 100  # 限制多少步还没分出胜负，则平手(与hoho_utils中的RESTRICT_ROUND_NUM相同)
match_count = 0

class Controller(chess.Controller):
	
	def __init__(self, board, history_winner = None):
		super(Controller, self).__init__(board)
		self.hoho_startup(win_player = history_winner)

	def onmouseup(self, ev):
		if self.dragging_chess is None: return
		x, y = ev.x.data(), ev.y.data()
		i2, j2 = self.chess_board.plate.pixel_to_nearest_pos(x, y)
		px, py = self.chess_board.plate.pos_to_pixel(i2, j2)
		# print(f'hoho: mouseup=({i2}, {j2}), x={x}, y={y}')
		near = chess._distance(x, y, px, py) < self.chess_board.setting.chess_size
		succ = False
		if near:
			succ, eaten = self._move_chess_to(self.dragging_chess, i2, j2)
		self._move_chess_img(self.dragging_chess, x, y)
		if succ:
			if (eaten is not None) and (eaten.type=='King'):
				javascript.alert("红方胜出!")
				self.restart()
				return
			self.player = 'Black'
		self.dragging_chess = None
		if self.player=='Black':
			self.blacks_turn()

	def blacks_turn(self):
		self.round_count = self.round_count + 1
		if self.round_count >= HOHO_RESTRICT_ROUND_NUM:
			if should_restart(match_count):
				restart_app()
			else:
				self.restart()
				self.hoho_startup()
			return

		spinner.show()
		self.chess_board.rotate_board()
		move_dict = auto_move_remote(self.chess_board, self.round_count)
		move_black = move_dict.get('Black')
		# print(f'hoho: Black move: {move_black}')
		self.chess_board.rotate_board()
		spinner.hide()
		if move_black is None:
			# javascript.alert("红方胜出!")
			if should_restart(match_count):
				restart_app(winner = 'Red')
			else:
				self.restart()
				self.hoho_startup('Red')
			return

		i1,j1,i2,j2 = move_black

		i1,j1,i2,j2 = 8-i1,9-j1,8-i2,9-j2
		chess = self.chess_board.board_map[(i1,j1)]
		succ, eaten = self._move_chess_to(chess, i2, j2)
		assert succ, 'black move is illegal!'
		px, py = self.chess_board.plate.pos_to_pixel(i1, j1)
		self._move_chess_img(chess, px, py)
		if (eaten is not None) and (eaten.type=='King'):
			# javascript.alert("黑方胜出!")
			if should_restart(match_count):
				restart_app(winner = 'Black')
			else:
				self.restart()
				self.hoho_startup('Black')
			return

		self.player = 'Red'
		move_red = move_dict.get('Red')
		self.hoho_red_turn(move_red)


	def hoho_red_turn(self, move):
		if move is None:
			# javascript.alert("黑方胜出!")
			if should_restart(match_count):
				restart_app(winner = 'Black')
			else:
				self.restart()
				self.hoho_startup('Black')
			return

		time.sleep(0.1)

		i1, j1, i2, j2 = move
		chess = self.chess_board.board_map[(i1, j1)]
		succ, eaten = self._move_chess_to(chess, i2, j2)
		assert succ, 'red move is illegal!'
		px, py = self.chess_board.plate.pos_to_pixel(i1, j1)
		self._move_chess_img(chess, px, py)
		if (eaten is not None) and (eaten.type == 'King'):
			# javascript.alert('红方胜出！')
			if should_restart(match_count):
				restart_app(winner = 'Red')
			else:
				self.restart()
				self.hoho_startup('Red')
			return

		self.player = 'Black'
		self.blacks_turn()


	def hoho_startup(self, win_player=None):
		global match_count
		match_count = match_count + 1

		time.sleep(0.1)

		done = False
		res = None
		def callback(data):
			nonlocal res
			nonlocal done
			if 'error' in data:
				javascript.alert(data['error'])
				done = True
				return
				
			if data is None:
				return

			res = data
			done = True

		ajax.send(('Action!', self.round_count, win_player), callback)
		while not done:
			time.sleep(.1)

		move = res
		self.hoho_red_turn(move)
		

def _chess_moves(chess):
	moves = []
	if chess.type=='Rock':
		for x in range(chess.x+1,9):
			if chess.can_move_to(x, chess.y):
				moves.append((x, chess.y))
			else:
				break
		for x in range(chess.x-1,-1,-1):
			if chess.can_move_to(x, chess.y):
				moves.append((x, chess.y))
			else:
				break
		for y in range(chess.y+1, 10):
			if chess.can_move_to(chess.x, y):
				moves.append((chess.x, y))
			else:
				break
		for y in range(chess.y-1, -1, -1):
			if chess.can_move_to(chess.x, y):
				moves.append((chess.x, y))
			else:
				break
	elif chess.type=='Cannon':
		for x in range(0, 9):
			if chess.can_move_to(x, chess.y):
				moves.append((x, chess.y))
		for y in range(0, 10):
			if chess.can_move_to(chess.x, y):
				moves.append((chess.x, y))
	elif chess.type in ('Knight','Guard','Bishop','Pawn','King'):
		for dx,dy in chess.allowed_moves:
			if chess.can_move_to(chess.x+dx, chess.y+dy):
				moves.append((chess.x+dx, chess.y+dy))
		if chess.type=='King':
			for y in (7,8,9):
				if chess.can_move_to(chess.x, y):
					moves.append((chess.x, y))
	else:
		raise
	return moves

def _get_next_moves(board):
	chesses = [chess for _, chess in board.board_map.items()]
	move_to_board = {}
	for chess in chesses:
		if chess.player=='Black': continue
		for x, y in _chess_moves(chess):
			board_key = []
			for c in chesses:
				if c is chess:
					board_key.append((c.player, c.type, x, y))
				elif (c.x==x) and (c.y==y):
					pass
				else:
					board_key.append((c.player, c.type, c.x, c.y))
			board_key.sort()
			move_key = (chess.player, chess.type, chess.x, chess.y, x, y)
			move_to_board[move_key] = tuple(board_key)
	return move_to_board

def _reverse_boardkey(board_key):
	reversed = [('Red' if p=='Black' else 'Black', t, 8-x, 9-y) for p,t,x,y in board_key]
	reversed.sort()
	return tuple(reversed)

def _board_from_key(board_key):
	board = chess.ChessBoard()
	board.board_map = {}
	types = {'Rock':chess.Rock, 'Knight':chess.Knight, 'Bishop':chess.Bishop, 
				'Guard':chess.Guard, 'King':chess.King, 'Cannon':chess.Cannon, 'Pawn':chess.Pawn}
	for player, type, x, y in board_key:
		board.board_map[(x,y)] = types[type](board, player, x, y)
	return board

def _board_key(board):
	board_key = [(c.player, c.type, c.x, c.y) for _, c in board.board_map.items()]
	board_key.sort()
	return tuple(board_key)

def should_restart(match_count):
	return match_count > 10


class BoardNode:
	win_score = 10000
	def __init__(self, board, board_key=None, move_key=None, depth=0):
		self.board = board
		self.board_key = _board_key(board) if board_key is None else board_key
		self.move_key = move_key
		self.depth = depth
		self.children = []
		self.parents = []
		self.score = self._estimate_score()
		self.best_child = None

	def _estimate_score(self):
		has_r_king = False
		score_r = 0
		score_b = 0
		score_r_cross = 0
		score_b_cross = 0
		chess_scores = {'Rock':30, 'Knight':10, 'Bishop':3, 
				'Guard':3, 'King':1, 'Cannon':10, 'Pawn':1}
		cross_river_factor = 1.1
		attack_factor = 0.2
		for (x,y),c in self.board.board_map.items():
			if c.player=='Red':
				if y<5:
					score_r = score_r + chess_scores[c.type]
					if c.type=='King': has_r_king=True
				else:
					score_r_cross = score_r_cross + chess_scores[c.type]
			else:
				if y>4:
					score_b = score_b+ chess_scores[c.type]
				else:
					score_b_cross = score_b_cross + chess_scores[c.type]
		if not has_r_king:
			return -BoardNode.win_score
		r_attack = max(0, score_r_cross-score_b) * attack_factor
		b_attack = max(0, score_b_cross-score_r) * attack_factor
		score_r = (score_r + (score_r_cross*cross_river_factor)) + r_attack
		score_b = (score_b + (score_b_cross*cross_river_factor)) + b_attack
		return score_r - score_b

	def same_as_ancester(self, boardkey):
		for p in self.parents:
			if p.board_key==boardkey: return True
			if p.same_as_ancester(boardkey): return True
		return False

	def expand(self):
		assert self.best_child is None
		move_to_board = _get_next_moves(self.board)
		for movekey, boardkey in move_to_board.items():
			boardkey = _reverse_boardkey(boardkey)
			if self.same_as_ancester(boardkey): continue
			if boardkey in board_explorer.board_cache:
				board_node = board_explorer.board_cache[boardkey]
			else:
				board = _board_from_key(boardkey)
				board_node = BoardNode(board, boardkey, movekey, self.depth+1)
				board_explorer.board_cache[boardkey] = board_node
				heapq.heappush(board_explorer.heap, board_node)
			board_node.parents.append(self)
			self.children.append(board_node)
			board_node.update_parents()

	def __lt__(self, other):
		# 使headq优先扩展最有可能的路径
		if (self.depth<=1) and (other.depth<=1):
			return self.score < other.score
		if self.depth<=1: return True
		if other.depth<=1: return False
		return (self.score,self.depth) < (other.score,other.depth)

	def update_score(self):
		self.best_child = None
		for c in self.children:
			if (self.best_child is None) or (self.score < (-c.score)):
				self.score = -c.score
				self.best_child = c
		self.update_parents()

	def update_parents(self):
		p_score = -self.score
		for p in self.parents:
			if (p.best_child is None) or (p.score < p_score):
				p.score = p_score
				p.best_child = self
				p.update_parents()
			elif (p.best_child is self) and (p.score > p_score):
				p.update_score()

class BoardExplorer:
	def __init__(self, time_limit):
		self.board_cache = None
		self.heap = None
		self.time_limit = time_limit
		# print('hoho: BoardExplorer created!')

	def run(self, board):
		board_node = BoardNode(board)
		self.board_cache = {}
		self.heap = [board_node]
		start_time = time.time()
		explored = 0
		while True:
			if (time.time()-start_time) > self.time_limit: 
				# print(f'hoho: time_limit!')
				break
			if len(board_explorer.heap)==0: 
				print(f'explorer heap empty!')
				break
			node = heapq.heappop(board_explorer.heap)
			score0 = node.score
			if score0 in (-BoardNode.win_score, BoardNode.win_score):
				continue # winned or lossed
			node.expand()
			# print(f'\n--- exploring#{explored} depth:{node.depth} score:{score0}>{node.score} ---')
			# print(node.board.board_map_text())
			explored = explored + 1
			elapsed = time.time()-start_time
		print(f'{explored} nodes were explored in {round(elapsed*100)/100} seconds')
		self.board_cache = None
		self.heap = None
		# _dump_tree(board_node)
		return board_node

def _dump_tree(node):
	import math
	with open('tree.txt', 'w') as fp:
		def dump_node(n,B=''):
			move = '' if n.move_key is None else '_'.join([str(e) for e in n.move_key])
			fp.write(f' ({B}{move}{round(n.score*100)/100}'.replace('-','_'))
			if len(n.children)==0:
				fp.write(' _')
			else:
				for c in n.children:
					dump_node(c,'B' if c==n.best_child else '')
			fp.write(')')
		dump_node(node)

board_explorer = BoardExplorer(3)

def auto_move(board):
	board_node = board_explorer.run(board)
	if board_node.best_child is None:
		return None
	# print('board_node.score', board_node.score)
	# print('board_node.best_child.move_key', board_node.best_child.move_key)
	# print('board_node.best_child.score', board_node.best_child.score)

	# print(f'hoho: board_node: {board_node.board_key}')
	# print(f'hoho: move: {board_node.best_child.move_key}')
	# print(f'hoho: board_node\'s best_child: {board_node.best_child.board_key}')
	return board_node.best_child.move_key[2:]
	
def auto_move_remote(board, round_count):
	board_key = _board_key(board)
	done = False
	res = None
	def callback(data):
		# print(f'hoho: receive = {data}')
		nonlocal res
		nonlocal done
		if 'error' in data:
			javascript.alert(data['error'])
			done = True
			return
	
		if data is None:
			return

		res = data
		done = True
	# print(f'hoho: send = {board_key}')
	ajax.send((board_key, round_count), callback)
	while not done:
		time.sleep(.1)
	return res


def run_app():
	# print('hoho: auto_chess run_app()!') # 用javascript打印
	chess_board = chess.ChessBoard()
	javascript.document.body.appendChild(chess_board.elt())
	controller = Controller(chess_board)


# 为避免嵌套调用太深，适时将重新创建整个Controller
def restart_app(winner = None):
	print(f'hoho: restart_app!')

	chess_board = chess.ChessBoard()
	javascript.document.body.removeChild(javascript.document.getElementById("hoho_board"))
	javascript.document.body.insertBefore(chess_board.elt(), javascript.document.body.lastChild)
	Controller(chess_board, history_winner = winner)
