import numpy as np


BOARD_WIDTH = 9
BOARD_HEIGHT = 10
BOARD_POSITION_NUM = BOARD_WIDTH * BOARD_HEIGHT
# K：帅，A：仕，R：车，B：相，N：马，P：兵，C：炮/ 大写红方，小写黑方
INDEX_2_PIECES = 'KARBNPCkarbnpc' # 14 x 9 x 10
PIECES_2_INDEX = {INDEX_2_PIECES[i]: i for i in range(len(INDEX_2_PIECES))}
X_LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
X_LABELS_2_INDEX = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
Y_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



# 棋盘每个点的标签
def get_position_labels():
    labels_array = []

    for x in range(len(X_LABELS)):
        for y in range(len(Y_LABELS)):
            move = X_LABELS[x] + Y_LABELS[y]
            labels_array.append(move)
    return labels_array

# 所有走子动作，一共有2086个走法
def get_all_moves():
    moves = []

    # 士的走法
    mandarins_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                        'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    
    # 象的走法
    elephants_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                        'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                        'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                        'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    cols = len(X_LABELS)
    rows = len(Y_LABELS)
    for c in range(cols):
        for r in range(rows):
            # 这里假设起始点为（c, r）,以下计算能走的目标点
            destinations = [(t, r) for t in range(cols)] + \
                        [(c, t) for t in range(rows)] + \
                        [(c + a, r + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]] # 马走日

        
            for (c2, r2) in destinations:
                if (c, r) != (c2, r2) and c2 in range(cols) and r2 in range(rows):
                    move = X_LABELS[c] + Y_LABELS[r] + X_LABELS[c2] + Y_LABELS[r2]
                    moves.append(move)
            
    moves.extend(mandarins_labels)
    moves.extend(elephants_labels)

    return moves


BOARD_POSITION_ARRAY = np.array(get_position_labels()).reshape(BOARD_WIDTH, BOARD_HEIGHT).transpose()
ACTION_DIM = get_all_moves()

class GameBoard(object):

# 棋盘的数组表示：小写表示黑方，大写表示红方
# [
#     "rnbakabnr",
#     "         ",
#     " c     c ",
#     "p p p p p",
#     "         ",
#     "         ",
#     "P P P P P",
#     " C     C ",
#     "         ",
#     "RNBAKABNR"
# ]

# 棋盘的字符串表示：
# "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"
# “/”表示换行，数字表示棋子之间或棋子与边界之间的空格数量
# 这里会先打印红方（大写），所以由上往下行序号由0到9，由左到右边列序号为a到i，符合通用规则


    def __init__(self):
        self.reload()

    def reload(self):
        self.board_state_str = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board_str, action=None):
        if action != None:  # action应该是类似: i1c4这样的表示
            src = action[0:2]

            src_x = int(X_LABELS_2_INDEX[src[0]])
            src_y = int(src[1])

        board = board_str.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')

        for i, line in enumerate(board):
            if action != None:
                if i == src_y:
                    s = list(line)
                    s[src_x] = 'x'   # 标记一下落子开始点
                    line = ''.join(s)
            print(line)


    # 将空格转为相当数量的数字1，并返回棋盘数组
    @staticmethod
    def board_str_to_list1(board_str):
        board = board_str.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")
    
    # 将空格数字1转换为数量，并返回棋盘字符串
    @staticmethod
    def board_list1_to_str(board_list):
        board = "/".join(board_list)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")
        return board

    # 走子，更改state表示（字符串形式）
    @staticmethod
    def do_action_on_board(action, board_str):
        src = action[0:2]
        dst = action[2:4]
        src_x = int(X_LABELS_2_INDEX[src[0]])
        src_y = int(src[1])
        dst_x = int(X_LABELS_2_INDEX[dst[0]])
        dst_y = int(dst[1])

        board_positions = GameBoard.board_str_to_list1(board_str)
        board_lines = []
        for line in board_positions:
            board_lines.append(list(line))

        board_lines[dst_y][dst_x] = board_lines[src_y][src_x]  # 将src的子走到dst位置
        board_lines[src_y][src_x] = '1'    # 走子后，src位置赋值为空白

        board_positions[dst_y] = ''.join(board_lines[dst_y])
        board_positions[src_y] = ''.join(board_lines[src_y])

        board = GameBoard.board_list1_to_str(board_lines)

        return board

    @staticmethod
    def check_bounds(toY, toX):
        if toY < 0 or toX < 0:
            return False

        if toY >= BOARD_HEIGHT or toX >= BOARD_WIDTH:
            return False

        return True

    # player能否走到棋子位置piece上（即吃掉棋子piece）
    @staticmethod
    def can_moveto_for_player(piece, player):
        if piece.isalpha():
            if player == 'r':
                if piece.islower():
                    return True
                else:
                    return False
            elif player == 'b':
                if piece.isupper():
                    return True
                else:
                    return False

        else:
            return True

    # 返回当前棋盘状态下所有合法走子
    @staticmethod
    def get_legal_moves(board_str, current_player):
        moves = []
        k_x = None  # 黑方将军（将）走子
        k_y = None

        K_x = None  # 红方将军（帅）走子
        K_y = None

        king_face2face = False

        board_positions = np.array(GameBoard.board_str_to_list1(board_str))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if board_positions[y][x].isalpha():
                    if board_positions[y][x] == 'r' and current_player == 'b': # 黑方车走子
                        toY = y
                        for toX in range(x - 1, -1, -1):  # 往左走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():   
                                if board_positions[toY][toX].isupper():  
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, BOARD_WIDTH): # 往右走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1): # 往上走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, BOARD_HEIGHT): # 往下走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif board_positions[y][x] == 'R' and current_player == 'r':  # 红方车走子
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():   
                                if board_positions[toY][toX].islower():   # 红方车吃掉对方的子
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, BOARD_WIDTH):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, BOARD_HEIGHT):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif (board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b':  # 黑方马走子
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):  
                                toY = y + 2 * i   # 向纵方向走“日字”
                                toX = x + 1 * j
                                # GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b')判断目标点是否为对方棋子
                                # board_positions[toY - i][x].isalpha() == False 判断马将要落子方向上是否有棋子挡住不能走
                                if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                                
                                toY = y + 1 * i   # 向横方向走“日字”
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif (board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'r':  # 红方马走子
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                                
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif (board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b':  # 黑方象走子
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i
                            # toY >= 5，限定黑方象只能在5~9行走（黑方在棋盘下方，从0行开始计算）
                            # board_positions[y + i // 2][x + i // 2].isalpha() == False，象前进方向上是否有子挡住不能走
                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and \
                               toY >= 5 and \
                               board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                            
                            toY = y + i
                            toX = x - i
                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and \
                               toY >= 5 and \
                               board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif (board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'r':  # 红方象走子
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and \
                               toY <= 4 and \
                               board_positions[y + i // 2][x + i // 2].isalpha() == False:
                               
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and \
                               toY <= 4 and \
                               board_positions[y + i // 2][x - i // 2].isalpha() == False:
                               
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif board_positions[y][x] == 'a' and current_player == 'b':  # 黑方士走子
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i
                            # toY >= 7 and toX >= 3 and toX <= 5， 限定黑方士在棋盘可移动坐标范围
                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and \
                               toY >= 7 and \
                               toX >= 3 and \
                               toX <= 5:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and \
                               toY >= 7 and \
                               toX >= 3 and \
                               toX <= 5:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif board_positions[y][x] == 'A' and current_player == 'r':  # 红方士走子
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and \
                               toY <= 2 and \
                               toX >= 3 and \
                               toX <= 5:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and \
                               GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and \
                               toY <= 2 and \
                               toX >= 3 and \
                               toX <= 5:
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif board_positions[y][x] == 'k': 
                        k_x = x
                        k_y = y

                        if current_player == 'b':  # 黑方将走子
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign  # i为0表示横向移
                                    toX = x + j * sign  # j为0表示纵向移

                                    if GameBoard.check_bounds(toY, toX) and \
                                       GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b') and \
                                       toY >= 7 and \
                                       toX >= 3 and \
                                       toX <= 5:
                                        moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif board_positions[y][x] == 'K':
                        K_x = x
                        K_y = y

                        if(current_player == 'r'):  # 红方帅走子
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and \
                                       GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r') and \
                                       toY <= 2 and \
                                       toX >= 3 and \
                                       toX <= 5:
                                        moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])
                    elif board_positions[y][x] == 'c' and current_player == 'b':  # 黑方炮走子
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):  # 往左方向走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha(): # 若目标点有子
                                    hits = True
                                else:
                                    moves.append(m)  
                            else:   # 循环的上一次扫描时发现有子挡在前进方向上
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():  # 若目标点是对方的子，可以吃掉
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, BOARD_WIDTH): # 往右方向走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):  # 往上方向走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, BOARD_HEIGHT):  # 往下方向走
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break
                    elif board_positions[y][x] == 'C' and current_player == 'r':  # 红方炮走子
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, BOARD_WIDTH):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, BOARD_HEIGHT):
                            m = BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX]
                            if hits == False:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break
                    elif board_positions[y][x] == 'p' and current_player == 'b':  # 黑方卒走子
                        toY = y - 1
                        toX = x

                        if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b'):
                            moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                        if y < 5: # 已经过河
                            toY = y
                            toX = x + 1 # 可以往右横走
                            if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b'):
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                            toX = x - 1 # 可以往左横走
                            if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'b'):
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                    elif board_positions[y][x] == 'P' and current_player == 'r':  # 红方兵走子
                        toY = y + 1
                        toX = x

                        if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r'):
                            moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r'):
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

                            toX = x - 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.can_moveto_for_player(board_positions[toY][toX], 'r'):
                                moves.append(BOARD_POSITION_ARRAY[y][x] + BOARD_POSITION_ARRAY[toY][toX])

        if K_x != None and k_x != None and K_x == k_x: # 当黑方将和红方帅面对面
            king_face2face = True
            for i in range(K_y + 1, k_y, 1):  # K_y 是红方的，序号肯定比k_y小
                if board_positions[i][K_x].isalpha():  # 判断黑方将和红方帅之间有没其他子阻挡
                    king_face2face = False

        if king_face2face == True:
            if current_player == 'b':  # 直杀对方将（帅）
                moves.append(BOARD_POSITION_ARRAY[k_y][k_x] + BOARD_POSITION_ARRAY[K_y][K_x])
            else:
                moves.append(BOARD_POSITION_ARRAY[K_y][K_x] + BOARD_POSITION_ARRAY[k_y][k_x])

        return moves



if __name__ == '__main__':
    gb = GameBoard()
    GameBoard.print_borad(gb.board_state_str)
    print(GameBoard.get_legal_moves(gb.board_state_str, 'b'))