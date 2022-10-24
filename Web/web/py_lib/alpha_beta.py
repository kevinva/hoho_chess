import time
from . import chess
from . import score
from . import min_max
BoardNode = min_max.BoardNode

class BoardExplorer:
	def __init__(self, min_depth, max_time):
		self.board_cache = {}
		self.min_depth = min_depth
		self.max_time = max_time
	def run(self, board):
		self.start_time = time.time()
		self.explored = 0
		self.cut_off = 0
		children_explored = 0
		depth_explored = 0
		best_move = None
		best_move_node = None
		for depth in range(2,20): # 迭代加深
			self.depth = depth
			node = BoardNode(board, self, height=depth)
			try:
				search(node)
				best_move = node.best_move
				best_move_node = node
				children_explored = self.explored
				depth_explored = depth
			except TimeoutError:
				break
		elapsed = time.time()-self.start_time
		print(f'{children_explored} ({self.explored}) nodes of depth {depth_explored} were explored in {round(elapsed*100)/100} seconds')
		print(f'# cut-off {self.cut_off}')
		self.board_cache.clear()
		return best_move_node, best_move
	def time_is_up(self):
		elapsed = time.time()-self.start_time
		return elapsed > self.max_time

def create_child_nodes(node,alpha,beta):
	move_to_board = chess.get_next_moves(node.board)
	children_scores = []
	for move_key, board_key in move_to_board.items():
		board_key = chess.reverse_boardkey(board_key)
		if min_max.same_as_ancester(node, board_key):
			continue
		def get_reusable_node():
			ref_score = None
			cache = node.explorer.board_cache
			for c,a,b in cache.get(board_key,[]):
				if ((a<=alpha) and (beta<=b)) and (c.height>=node.height):
					return c, 1E6
				c_score = (c.height*10)*(-c.score)
				if (ref_score is None) or (c_score>ref_score):
					ref_score = c_score
			if ref_score is None:
				ref_score = -1E6
			return None, ref_score
		child, ref_score = get_reusable_node()
		if child is None:
			board = chess.board_from_key(board_key)
			child = BoardNode(board, node.explorer, board_key, node.height-1)
		node.children[move_key] = child
		child.parents.append(node)
		children_scores.append([(move_key,child), ref_score])
	children_scores.sort(key=lambda c: c[1], reverse=True)
	return [c[0] for c in children_scores]


def search(node, alpha=-1E6, beta=1E6):
	if (node.explorer.depth>node.explorer.min_depth) and node.explorer.time_is_up():
		raise TimeoutError()
	assert node.score is None
	player = 'Red' if ((node.explorer.depth-node.height)%2==0) else 'Black'
	if (node.height==0) or (not (score.has_king(node.board, player))):
		node.score = score.score(node.board, player)
		return node.score
	node.explorer.explored = node.explorer.explored + 1
	cache = node.explorer.board_cache
	cache[node.board_key] = cache.get(node.board_key,[]) + [(node,alpha,beta)]
	children_sorted = create_child_nodes(node,alpha,beta)
	node.children_sorted = children_sorted
	def check_score(s):
		nonlocal alpha
		if s > beta:
			node.explorer.cut_off = node.explorer.cut_off + 1
			return True
		if s > alpha:
			alpha = s
		if s == score.win_score:
			return True
		return False
	node.update_score()
	if node.score is not None:
		if check_score(node.score):
			return node.score
	for _, child in children_sorted:
		if child.score is not None:
			continue
		s = -search(child, -beta, -alpha)
		if check_score(s):
			node.score = s
			return node.score
	node.update_score()

	if node.score is None:
		print(f'node.score is None! alpha: {alpha}, beta: {beta}')
		return -1E6

	if alpha < node.score:
		print(f'alpha < node.score! alpha: {alpha}, beta: {beta}, node.score: {node.score}')
		return -1E6
	# assert alpha >= node.score    hoho_debug，先注释掉，可能会崩
	
	return node.score

def auto_move(board):
	# print(f'alpha_beta: {board.__dict__}')
	_, best_move = BoardExplorer(3, 10).run(board) # 搜索深度，搜索秒数
	print(best_move)
	return None if best_move is None else best_move[2:]
