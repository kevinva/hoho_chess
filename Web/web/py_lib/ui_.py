
import math
from . import chess

def html(tag, attrs=None, style=None):
	elt = javascript.document.createElement(tag)
	if attrs is not None:
		for k,v in attrs.items():
			elt.setAttribute(k,v)
	if style is not None:
		style = ';'.join([f'{k}:{v}' for k,v in style.items()])
		elt.setAttribute('style',style)
	return elt

class SettingLarge:
	offset_x = 29
	offset_y = 29-3
	grid_size_x = 59
	grid_size_y = 61
	plate_size = 535
	chess_size = 55

def _distance(x1, y1, x2, y2):
	return math.sqrt(((x1-x2)**2) + ((y1-y2)**2))

class Plate:
	def __init__(self, setting):
		self.setting = setting
		style = {'opacity':0.65, 'width':setting.plate_size,
				'position':'absolute','left':'0px','top':'0px'}
		self.elt = html('img', {'src':'chess-img/plate.png', 'draggable':'false'}, style=style)
	def pos_to_pixel(self, i, j):
		x = self.setting.offset_x + (i * self.setting.grid_size_x)
		y = self.setting.offset_y + ((9 - j) * self.setting.grid_size_y)
		return x, y	
	def pixel_to_nearest_pos(self, x, y):
		index = None
		min_dis = None
		for i in range(9):
			for j in range(10):
				px, py = self.pos_to_pixel(i, j)
				dis = _distance(x, y, px, py)
				if (index is None) or (dis < min_dis):
					index, min_dis = (i,j), dis
		return index

class ChessBoard(chess.ChessBoard):
	def __init__(self, setting=None):
		if setting is None:
			setting = SettingLarge()
		self.setting = setting
		self._elt = None
		self.init_board()

	def _refresh_elt(self):
		while self._elt.lastChild.data() is not None:
			self._elt.removeChild(self._elt.lastChild)
		self.plate = Plate(self.setting)
		self._elt.appendChild(self.plate.elt)
		self.img_map = {}
		size = (self.setting.chess_size / 2)
		for (i,j), chess in self.board_map.items():
			x,y = self.plate.pos_to_pixel(i,j)
			style = {'position':'absolute','left':f'{x-size}px','top':f'{y-size}px','width':f'{self.setting.chess_size}px','opacity':0.95, 'z-index':'0'}
			img = html('img', {'src':f'chess-img/{chess.player}_{chess.type.lower()}.png','draggable':'false'}, style)
			self._elt.appendChild(img)
			self.img_map[(i,j)]=img

	def elt(self):
		if self._elt is None:
			self._elt = html('div')
			self._refresh_elt()
		return self._elt

class Controller:
	def __init__(self, chess_board):
		self.chess_board = chess_board
		elt = chess_board.elt()
		elt.bind('mouseup', self.onmouseup)
		elt.bind('mousedown', self.onmousedown)
		elt.bind('mousemove', self.onmousemove)
		self.restart()

	def restart(self):
		self.dragging_chess = None
		self.player = 'Red'
		self.chess_board.init_board()
		self.chess_board._refresh_elt()

	def onmousedown(self, ev):
		if self.dragging_chess is not None:
			return
		x, y = ev.x.data(), ev.y.data()
		i, j = self.chess_board.plate.pixel_to_nearest_pos(x, y)
		px, py = self.chess_board.plate.pos_to_pixel(i, j)
		if _distance(x, y, px, py) > self.chess_board.setting.chess_size:
			return
		if (i,j) not in self.chess_board.board_map:
			return
		chess = self.chess_board.board_map[(i,j)]
		if chess.player!=self.player:
			return False
		self.dragging_chess = chess
		img = self.chess_board.img_map[(i,j)]
		setattr(img.style, 'z-index', '1')

	def _can_move_chess_to(self, chess, i2, j2):
		if self.player=='Black':
			i2,j2 = 8-i2, 9-j2
			self.chess_board.rotate_board()
		succ = chess.can_move_to(i2,j2)
		if self.player=='Black':
			self.chess_board.rotate_board()
		return succ

	def _move_chess_to(self, chess, i2, j2):
		if not self._can_move_chess_to(chess, i2, j2):
			return False, None
		i1, j1 = chess.x, chess.y
		chess.x, chess.y = i2, j2
		img = self.chess_board.img_map[(i1,j1)]
		del self.chess_board.board_map[(i1,j1)]
		del self.chess_board.img_map[(i1,j1)]
		captured = None
		if (i2,j2) in self.chess_board.board_map:
			captured = self.chess_board.board_map[(i2,j2)]
			del self.chess_board.board_map[(i2,j2)]
			img0 = self.chess_board.img_map[(i2,j2)]
			del self.chess_board.img_map[(i2,j2)]
			self.chess_board.elt().removeChild(img0)
		self.chess_board.board_map[(i2,j2)] = chess
		self.chess_board.img_map[(i2,j2)] = img
		return True, captured

	def _move_chess_img(self, chess, x0, y0, animation_time=.5, animation_frames=25):
		i, j = chess.x, chess.y
		img = self.chess_board.img_map[(i,j)]
		px, py = self.chess_board.plate.pos_to_pixel(i, j)
		size = self.chess_board.setting.chess_size/2
		player = self.player
		dragging_chess = self.dragging_chess
		self.player = None
		self.dragging_chess = None
		import time
		frames = int(animation_frames*animation_time)
		for i in range(frames):
			x = x0+((px-x0)*((i+1)/frames))
			y = y0+((py-y0)*((i+1)/frames))
			img.style.left = f'{x-size}px'
			img.style.top = f'{y-size}px'
			time.sleep(1/animation_frames)
		assert x==px, (x,px)
		setattr(img.style, 'z-index', '0')
		self.player = player
		self.dragging_chess = dragging_chess

	def onmouseup(self, ev):
		if self.dragging_chess is None: return
		x, y = ev.x.data(), ev.y.data()
		i2, j2 = self.chess_board.plate.pixel_to_nearest_pos(x, y)
		px, py = self.chess_board.plate.pos_to_pixel(i2, j2)
		near = _distance(x, y, px, py) < self.chess_board.setting.chess_size
		if near:
			succ, captured = self._move_chess_to(self.dragging_chess, i2, j2)
			if succ:
				if (captured is not None) and (captured.type=='King'):
					javascript.alert(f"{'红' if self.player=='Red' else '黑'}方胜出!")
					self.restart()
					return
				self.player = 'Red' if self.player=='Black' else 'Black'
		self._move_chess_img(self.dragging_chess, x, y)
		self.dragging_chess = None

	def onmousemove(self, ev):
		if self.dragging_chess is None: return
		i,j = self.dragging_chess.x, self.dragging_chess.y
		img = self.chess_board.img_map[(i,j)]
		size = self.chess_board.setting.chess_size/2
		img.style.left = f'{ev.x.data()-size}px'
		img.style.top = f'{ev.y.data()-size}px'


def run_app():
	chess_board = ChessBoard()
	javascript.document.body.appendChild(chess_board.elt())
	Controller(chess_board)
	