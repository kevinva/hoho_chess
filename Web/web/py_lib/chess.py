
class ChessMan:
	pos_range=(0,0,8,9)
	def __init__(self, board, player, type, x, y):
		self.board = board
		self.player = player
		self.type = type
		self.x = x
		self.y = y

	def clone(self):
		return type(self)(self.board, self.type, self.x, self.y)

	def can_move_to(self, x, y):
		if (x<self.pos_range[0]) or (y<self.pos_range[1]): return False
		if (x>self.pos_range[2]) or (y>self.pos_range[3]): return False
		chess = self.board.board_map.get((x, y))
		if (chess is not None) and (chess.player==self.player): 
			return False
		dx = x-self.x
		dy = y-self.y
		if (dx==0) and (dy==0): return False
		if not hasattr(self, 'allowed_moves'): return True
		if (dx, dy) in self.allowed_moves: return True
		return False

class King(ChessMan):
	def __init__(self, board, player, x, y):
		super(King, self).__init__(board, player, 'King', x, y)
		self.allowed_moves=((-1,0),(1,0),(0,-1),(0,1))
		self.pos_range=(3,0,5,2)
	def can_move_to(self, x, y):
		if super(King, self).can_move_to(x, y):
			return True
		if (self.x==x) and (self.y<y):
			cs = self.board._chesses_between(self.x, self.y, x, y)
			if cs==0:
				dest = self.board.board_map.get((x, y))
				if dest is not None:
					return dest.type == self.type
		return False

class Rock(ChessMan):
	def __init__(self, board, player, x, y):
		super(Rock, self).__init__(board, player, 'Rock', x, y)
	def can_move_to(self, x, y):
		if (self.x!=x) and (self.y!=y): return False
		if not super(Rock, self).can_move_to(x, y):
			return False
		if self.board._chesses_between(self.x, self.y, x, y) > 0:
			return False
		dest = self.board.board_map.get((x,y))
		if dest is None:
			return True
		return dest.player != self.player

class Cannon(ChessMan):
	def __init__(self, board, player, x, y):
		super(Cannon, self).__init__(board, player, 'Cannon', x, y)
	def can_move_to(self, x, y):
		if (self.x!=x) and (self.y!=y): return False
		if not super(Cannon, self).can_move_to(x, y): 
			return False
		cs = self.board._chesses_between(self.x, self.y, x, y)
		dest = self.board.board_map.get((x,y))
		if cs==0:
			return dest is None
		elif cs==1:
			return (dest is not None) and (dest.player != self.player)
		return False

class Knight(ChessMan):
	def __init__(self, board, player, x, y):
		super(Knight, self).__init__(board, player, 'Knight', x, y)
		self.allowed_moves=((-2,-1),(-2,1),(-1,-2),(-1,2),(2,-1),(2,1),(1,-2),(1,2))
	def can_move_to(self, x, y):
		if not super(Knight, self).can_move_to(x, y):
			return False
		dx = x-self.x
		dy = y-self.y
		if dx == (-2): block = (self.x-1,self.y)
		elif dx == 2: block = (self.x+1,self.y)
		elif dy == (-2): block = (self.x,self.y-1)
		elif dy == 2: block = (self.x,self.y+1)
		else: return False
		return block not in self.board.board_map

class Guard(ChessMan):
	def __init__(self, board, player, x, y):
		super(Guard, self).__init__(board, player, 'Guard', x, y)
		self.allowed_moves=((-1,-1),(-1,1),(1,-1),(1,1))
		self.pos_range=(3,0,5,2)

class Bishop(ChessMan):
	def __init__(self, board, player, x, y):
		super(Bishop, self).__init__(board, player, 'Bishop', x, y)
		self.allowed_moves=((-2,-2),(-2,2),(2,-2),(2,2))
		self.pos_range=(0,0,8,4)
	def can_move_to(self, x, y):
		if not super(Bishop, self).can_move_to(x, y):
			return False
		block = (self.x+((x-self.x)//2), self.y+((y-self.y)//2))
		return block not in self.board.board_map

class Pawn(ChessMan):
	def __init__(self, board, player, x, y):
		super(Pawn, self).__init__(board, player, 'Pawn', x, y)
		self.allowed_moves=((0,1),(-1,0),(1,0))
		self.pos_range=(0,3,8,9)
	def can_move_to(self, x, y):
		if not super(Pawn, self).can_move_to(x, y):
			return False
		return (self.y>=5) or (x==self.x)

class ChessBoard:
	def __init__(self):
		self.init_board()
	def init_board(self):
		self.board_map = {}
		def init_red():
			cs = []
			cs.append(((0,0),Rock))
			cs.append(((1,0),Knight))
			cs.append(((2,0),Bishop))
			cs.append(((3,0),Guard))
			cs.append(((4,0),King))
			cs.append(((5,0),Guard))
			cs.append(((6,0),Bishop))
			cs.append(((7,0),Knight))
			cs.append(((8,0),Rock))
			cs.append(((1,2),Cannon))
			cs.append(((7,2),Cannon))
			for i in range(5):
				cs.append(((i*2,3),Pawn))
			for (x,y),cm_type in cs:
				self.board_map[(x,y)] = cm_type(self, 'Red', x, y)
		init_red()
		self.rotate_board()
		init_red()

	def rotate_board(self):
		board_map = [((i,j),chess) for (i,j),chess in self.board_map.items()]
		self.board_map.clear()
		for (i,j),chess in board_map:
			chess.x = 8-i
			chess.y = 9-j
			chess.player = 'Red' if chess.player=='Black' else 'Black'
			self.board_map[(chess.x,chess.y)] = chess

	# for debugging
	def board_map_text(self):
		name = {'Red-Pawn':'兵','Black-Pawn':'卒',
				'Red-Bishop':'相','Black-Bishop':'象',
				'Red-Guard':'仕','Black-Guard':'士',
				'Red-Cannon':'炮','Black-Cannon':'炮',
				'Red-Knight':'马','Black-Knight':'马',
				'Red-Rock':'车','Black-Rock':'车',
				'Red-King':'帅','Black-King':'将'}
		text = [['　' for x in range(9)] for y in range(10)]
		for (i,j),chess in self.board_map.items():
			text[9-j][i] = name[f'{chess.player}-{chess.type}']
		return '\n'.join([''.join([c for c in l]) for l in text])

	def _chesses_between(self, x1, y1, x2, y2):
		if (x1!=x2) and (y1!=y2):
			return 0
		if x1==x2:
			cs = [self.board_map.get((x1,y)) 
				for y in (range(y1+1,y2) if y1<y2 else range(y2+1,y1))]
		if y1==y2:
			cs = [self.board_map.get((x,y1)) 
				for x in (range(x1+1,x2) if x1<x2 else range(x2+1,x1))]
		return len([c for c in cs if c is not None])


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

def get_next_moves(board):
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

def reverse_boardkey(board_key):
	reversed = [('Red' if p=='Black' else 'Black', t, 8-x, 9-y) for p,t,x,y in board_key]
	reversed.sort()
	return tuple(reversed)

def board_from_key(board_key):
	board = ChessBoard()
	board.board_map = {}
	types = {'Rock':Rock, 'Knight':Knight, 'Bishop':Bishop, 
				'Guard':Guard, 'King':King, 'Cannon':Cannon, 'Pawn':Pawn}
	for player, type, x, y in board_key:
		board.board_map[(x,y)] = types[type](board, player, x, y)
	return board

def board_key(board):
	key = [(c.player, c.type, c.x, c.y) for _, c in board.board_map.items()]
	key.sort()
	return tuple(key)
