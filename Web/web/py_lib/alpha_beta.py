import time
from . import chess
from . import score
from . import min_max
BoardNode = min_max.BoardNode

class BoardExplorer:
	def __init__(self, min_depth, max_time):
		self.min_depth = min_depth
		self.max_time = max_time
	def run(self, board):
		self.start_time = time.time()
		self.explored = 0
		self.cut_off = 0 # 剪枝数目
		cur_depth = 0
		best_move = None
		cur_node = None
		cur_explored = 0
		for depth in range(self.min_depth,20): # 迭代加深
			self.depth = depth
			node = BoardNode(board, self, height=depth)
			node.alpha, node.beta = -1E6, 1E6
			try:
				search(node)
				best_move = node.best_move
				cur_node,cur_explored,cur_depth = node,self.explored,depth
			except TimeoutError:
				break
		elapsed = time.time()-self.start_time
		print(f'搜索节点数:{self.explored}(有效:{cur_explored}), 深度 {cur_depth}, 用时 {round(elapsed*100)/100}, 剪枝 {self.cut_off}')
		return cur_node, best_move
	def time_is_up(self):
		elapsed = time.time()-self.start_time
		return elapsed > self.max_time

def create_child_nodes(node):
	move_n_board = chess.get_next_moves(node.board)
	children = []
	for move_key, board_key in move_n_board.items():
		board_key = chess.reverse_boardkey(board_key)
		board = chess.board_from_key(board_key)
		child = BoardNode(board, node.explorer, board_key, node.height-1)
		child.children = child.parents = None # not used
		children.append([move_key, child])
	return children

def search(node):
	if (node.explorer.depth>node.explorer.min_depth) and node.explorer.time_is_up():
		raise TimeoutError()
	assert node.score is None
	player,opponent = 'Red','Black'
	assert score.has_king(node.board, opponent), node.board.board_map_text()
	if (node.height==0) or (not score.has_king(node.board, player)):
		node.score = score.score(node.board, player)
		return
	node.explorer.explored += 1
	node.score = -1E6
	children_sorted = create_child_nodes(node)
	for move, child in children_sorted:
		child.alpha, child.beta = -node.beta, min(-node.alpha, -node.score)
		search(child)
		if node.score < -child.score:
			node.score = -child.score
			node.best_move = move
		if node.score >= node.beta:
			node.best_move = None
			node.explorer.cut_off += 1
			break

be = BoardExplorer(4, 20)
def auto_move(board):
	_, best_move = be.run(board) # 搜索深度，搜索秒数
	print(best_move)
	return None if best_move is None else best_move[2:]
