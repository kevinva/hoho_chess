from . import ui_
from . import spinner_
from . import chess

HOHO_RESTRICT_ROUND_NUM = 100  # 限制多少步还没分出胜负，则平手(与hoho_utils中的RESTRICT_ROUND_NUM相同)


class Controller(ui_.Controller):

	def __init__(self, board):
		super(Controller, self).__init__(board)
		self.round_num = 0
		self.hoho_reset()

	def onmouseup(self, ev):
		print(f'hoho: onmouseup!')
		
		if self.dragging_chess is None: return
		x, y = ev.x.data(), ev.y.data()
		i2, j2 = self.chess_board.plate.pixel_to_nearest_pos(x, y)
		px, py = self.chess_board.plate.pos_to_pixel(i2, j2)
		near = ui_._distance(x, y, px, py) < self.chess_board.setting.chess_size
		succ = False
		if near:
			succ, captured = self._move_chess_to(self.dragging_chess, i2, j2)
		self._move_chess_img(self.dragging_chess, x, y)
		if succ:
			if (captured is not None) and (captured.type=='King'):
				javascript.alert("红方胜出!")
				self.restart()
				return
			self.player = 'Black'
		self.dragging_chess = None
		if self.player=='Black':
			self.blacks_turn()

	def blacks_turn(self):
		spinner_.show()
		self.chess_board.rotate_board()
		# move = auto_move(self.chess_board)
		try:
			from . import ajax_
			board_key = chess.board_key(self.chess_board) # board_key 可变为 JSON
			move = ajax_.rpc.rpc_auto_move(board_key)
		except RuntimeError as ex:
			javascript.alert(str(ex))
			return
		self.chess_board.rotate_board()
		spinner_.hide()
		if move is None:
			javascript.alert("红方胜出!")
			self.restart()
			return

		print(f'blacks_ture: {move}')

		i1,j1,i2,j2 = move
		i1,j1,i2,j2 = 8-i1,9-j1,8-i2,9-j2
		chess1 = self.chess_board.board_map[(i1,j1)]
		succ, captured = self._move_chess_to(chess1, i2, j2)
		assert succ
		px, py = self.chess_board.plate.pos_to_pixel(i1, j1)
		self._move_chess_img(chess1, px, py)
		if (captured is not None) and (captured.type=='King'):
			javascript.alert("黑方胜出!")
			self.restart()
			return
		self.player = 'Red'

	def hoho_reset(self, winner = None):
		self.round_num = 0


def run_app():
	chess_board = ui_.ChessBoard()
	javascript.document.body.appendChild(chess_board.elt())
	Controller(chess_board)

