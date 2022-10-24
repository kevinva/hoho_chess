import time
from . import chess
from . import score

class BoardNode:
	def __init__(self, board, explorer, board_key=None, height=0):
		self.board = board
		self.explorer = explorer
		self.board_key = chess.board_key(board) if board_key is None else board_key
		self.height = height
		self.children = {}
		self.parents = []
		self.score = None
		self.best_move = None

	def update_score(self):
		self.best_move = None
		for move, child in self.children.items():
			if child.score is None: continue
			if (self.best_move is None) or (self.score < (-child.score)):
				self.score = -child.score
				self.best_move = move
		self.update_parents()

	def update_parents(self):
		for p in self.parents:
			p.update_score()

class BoardExplorer:
	def __init__(self, depth):
		self.board_cache = None
		self.depth = depth
	def run(self, board):
		self.board_cache = {}
		self.start_time = time.time()
		self.explored = 0
		node = BoardNode(board, self, height=self.depth)
		search(node)
		elapsed = time.time()-self.start_time
		print(f'{self.explored} nodes of depth {self.depth} were explored in {round(elapsed*100)/100} seconds')
		self.board_cache = None
		return node.best_move

def same_as_ancester(node, boardkey):
	for p in node.parents:
		if p.board_key==boardkey: return True
		if same_as_ancester(p, boardkey): return True
	return False

def create_child_nodes(node):
	move_to_board = chess.get_next_moves(node.board)
	for move_key, board_key in move_to_board.items():
		board_key = chess.reverse_boardkey(board_key)
		if same_as_ancester(node, board_key):
			continue
		child = None
		# 使用搜索过的结点，前提是它的搜索深度不少于当前孩子结点
		for height in range(node.explorer.depth, node.height-2, -1):
			assert height >= (node.height-1)
			if (board_key, height) in node.explorer.board_cache:
				child = node.explorer.board_cache[(board_key, height)]
				break
		if child is None:
			board = chess.board_from_key(board_key)
			child = BoardNode(board, node.explorer, board_key, node.height-1)
		node.children[move_key] = child
		child.parents.append(node)

def search(node):
	assert node.score is None
	player = 'Red' if ((node.explorer.depth-node.height)%2==0) else 'Black'
	if (node.height==0) or (not (score.has_king(node.board, player))):
		node.score = score.score(node.board, player)
		# node.update_parents()
		return node.score
	node.explorer.explored = node.explorer.explored + 1
	node.explorer.board_cache[(node.board_key, node.height)] = node
	create_child_nodes(node)
	for move_key, child in node.children.items():
		if child.score is None:
			search(child)
	node.update_score()
	return node.score

def auto_move(board):
	best_move = BoardExplorer(4).run(board)
	return None if best_move is None else best_move[2:]
