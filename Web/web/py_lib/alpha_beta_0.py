import time
from . import chess
from . import score
from . import min_max
BoardNode = min_max.BoardNode

class BoardExplorer:
	def __init__(self, min_depth, max_time):
		self.min_depth = min_depth
		self.max_time = max_time
		self.board_cache = {}
	def run(self, board):
		self.start_time = time.time()
		self.explored = 0
		self.computed = 0
		self.cached = 0
		self.cut_off = 0 # 剪枝数目
		cur_depth = 0
		best_move = None
		cur_node = None
		cur_explored = 0
		for depth in range(2,20): # 迭代加深
			self.depth = depth
			node = BoardNode(board, self, height=depth)
			node.alpha, node.beta = -1E6, 1E6
			try:
				search(node)
				best_move = node.best_move
				cur_node, cur_explored, cur_depth = node, self.explored, depth
			except TimeoutError:
				break
		elapsed = time.time()-self.start_time
		print(f'搜索节点数:{self.explored}(有效:{cur_explored}), 计算:{self.computed}, 缓存:{self.cached}, 深度 {cur_depth}, 用时 {round(elapsed*100)/100}')
		print(f'# 剪枝 {self.cut_off}')
		return cur_node, best_move
	def time_is_up(self):
		elapsed = time.time()-self.start_time
		return elapsed > self.max_time

# def _score_from_cache(node, board_key, update_move):
# 	for prev in node.explorer.board_cache.get(board_key, []):
# 		node.ref_max_score = max(node.ref_max_score, prev.score)
# 		if prev.height < node.height: continue
# 		# 旧节点上界较小，且已剪枝，分数没参考价值
# 		if prev.beta<node.beta and prev.beta<=prev.score<node.beta : continue
# 		# 旧节点下界较大，且已剪枝，分数没参考价值
# 		if prev.alpha>node.alpha and node.alpha<prev.score<=prev.alpha: continue
# 		if node.score is None or node.score<prev.score:
# 			node.score = prev.score
# 			node.best_move = None
# 			if node.score >= node.beta: 
# 				break
# 			if prev.best_move is not None:
# 				node.best_move = update_move(prev.best_move)			

# def _add_to_cache(node):
# 	if node.score <= node.alpha:
# 		node.best_move = None
# 	if node.board_key not in node.explorer.board_cache:
# 		node.explorer.board_cache[node.board_key] = []
# 	node.explorer.board_cache.get(node.board_key).append(node)

# def score_from_cache(node):
# 	node.ref_max_score = -1E6
# 	_score_from_cache(node, node.board_key, lambda move_key: move_key)
# 	def update_move_1(move_key):
# 		player, chess_type, x1, y1, x2, y2 = move_key
# 		player = ('Red' if player=='Black' else 'Black')
# 		return (player, chess_type, 8-x1, 9-y1, 8-x2, 9-y2)
# 	_score_from_cache(node, chess.reverse_boardkey(node.board_key), update_move_1)
# 	def update_move_2(move_key):
# 		player, chess_type, x1, y1, x2, y2 = move_key
# 		return (player, chess_type, 8-x1, y1, 8-x2, y2)
# 	_score_from_cache(node, chess.flip_boardkey(node.board_key), update_move_2)
# 	def update_move_3(move_key):
# 		player, chess_type, x1, y1, x2, y2 = move_key
# 		player = ('Red' if player=='Black' else 'Black')
# 		return (player, chess_type, x1, 9-y1, x2, 9-y2)
# 	_score_from_cache(node, chess.reverse_flip_boardkey(node.board_key), update_move_3)
# 	if node.score is not None:
# 		_add_to_cache(node)

def create_child_nodes(node):
	move_n_board = chess.get_next_moves(node.board)
	children_1 = []
	children_2 = []
	for move_key, board_key in move_n_board.items():
		board_key = chess.reverse_boardkey(board_key)
		board = chess.board_from_key(board_key)
		child = BoardNode(board, node.explorer, board_key, node.height-1)
		child.children = child.parents = None # not used
		# child.alpha, child.beta = -node.beta, min(-node.alpha, -node.score)
		# score_from_cache(child)
		if child.score is not None:
			children_1.append([move_key, child])
		else:
			children_2.append([move_key, child])
	# children_2.sort(key=lambda x: x[1].ref_max_score, reverse=True)
	return children_1 + children_2

def search(node):
	if (node.explorer.depth>node.explorer.min_depth) and node.explorer.time_is_up():
		raise TimeoutError()
	assert node.score is None
	player = 'Red' if ((node.explorer.depth-node.height)%2==0) else 'Black'
	if (node.height==0) or (not (score.has_king(node.board, player))):
		node.score = score.score(node.board, player)
		return
	node.explorer.explored += 1
	node.score = -1E6
	children_sorted = create_child_nodes(node)
	for move, child in children_sorted:
		child.alpha, child.beta = -node.beta, min(-node.alpha, -node.score)
		# if child.score is None:
		# 	score_from_cache(child)
		if child.score is not None:
			child.explorer.cached += 1
		else:
			search(child)
			if child.height>0:
				child.explorer.computed += 1
		if node.score < -child.score:
			node.score = -child.score
			node.best_move = move
		if node.score >= node.beta:
			node.best_move = None
			break
	# _add_to_cache(node)

be = BoardExplorer(4, 20)
def auto_move(board):
	_, best_move = be.run(board) # 搜索深度，搜索秒数
	print(best_move)
	return None if best_move is None else best_move[2:]
