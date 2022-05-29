from config import BOARD_WIDTH, BOARD_HEIGHT

# K：帅，A：仕，R：车，B：相，N：马，P：兵，C：炮/ 大写红方，小写黑方
pieces_order = 'KARBNPCkarbnpc' # 9 x 10 x 14
ind = {pieces_order[i]: i for i in range(14)}

labels_array = create_all_moves()  # 所有走法
labels_len = len(labels_array)
flipped_labels = flipped_moves_labels(labels_array)  # 所有走法的镜像
unflipped_index = [labels_array.index(x) for x in flipped_labels]  # hoho: 将走法翻转一下，又求回在未翻转数组下的序号，意义是啥？多此一举！

i2label = {i: val for i, val in enumerate(labels_array)}
label2i = {val: i for i, val in enumerate(labels_array)}


# 所有走子动作，一共有2086个走法
def create_all_moves():
    moves = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 士的走法
    mandarins_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                        'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    
    # 象的走法
    elephants_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                        'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                        'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                        'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    cols = len(letters)
    rows = len(numbers)
    for c in range(cols):
        for r in range(rows):
            # 这里假设起始点为（c, r）,以下计算能走的目标点
            destinations = [(t, r) for t in range(cols)] + \
                           [(c, t) for t in range(rows)] + \
                           [(c + a, r + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]] # 马走日

        
            for (c2, r2) in destinations:
                if (c, r) != (c2, r2) and c2 in range(cols) and r2 in range(rows):
                    move = letters[c] + numbers[r] + letters[c2] + numbers[r2]
                    moves.append(move)
            
    moves.extend(mandarins_labels)
    moves.extend(elephants_labels)

    return moves


def create_position_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]  # hoho: 前面不用reverse，这里也就不用8去减了！多此一举！
            labels_array.append(move)
#     labels_array.reverse()
    return labels_array

def create_position_labels_reverse():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()
    return labels_array


# 相当于以中间河为镜子，将走法翻转，如走d7e8则其镜像为d2e1
def flipped_moves_labels(param):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in param]


def get_pieces_count(state):
    count = 0
    for s in state:
        if s.isalpha():
            count += 1
    return count


def is_kill_move(state_prev, state_next):
    return get_pieces_count(state_prev) - get_pieces_count(state_next)



class GameBoard(object):
    BOARD_POS_NAME = np.array(create_position_labels()).reshape(9,10).transpose()

# 小写表示黑方，大写表示红方
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

    def __init__(self):
        # state如下图所示：从下往上扫描，“/”表示换上一行，数字表示棋子之间或棋子与边界之间的空白距离
        # 这里state会先打印红方（大写），所以由上往下行序号由0到9，由左到右边列序号为a到i，符合通用规则
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        # self.players = ["w", "b"]
        self.current_player = "w"   # w表示红方，b表示黑方
        self.restrict_round = 0

    def reload(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board, action = None):
        def string_reverse(string):
            # return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
            return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if(action != None):  # action应该是类似: i1c4这样的表示
            src = action[0:2]

            src_x = int(x_trans[src[0]])
            src_y = int(src[1])

        # board = string_reverse(board)
        board = board.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')
        # board = board.replace("/", "\n")
        print("  abcdefghi")
        for i,line in enumerate(board):
            if (action != None):
                if(i == src_y):
                    s = list(line)
                    s[src_x] = 'x'   # 标记一下落子开始点
                    line = ''.join(s)
            print(i,line)
        # print(board)

    # 走子，更改state表示（字符串形式）
    @staticmethod
    def sim_do_action(in_action, in_state):
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # GameBoard.print_borad(in_state)
        # print("sim_do_action : ", in_action)
        # print(dst_y, dst_x, src_y, src_x)
        # in_state为空格用空白字符表示的，这里要将空白字符转换为相当数量的数字1
        board_positions = GameBoard.board_to_pos_name(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)
        # print(lines.shape)
        # print(board_positions[src_y])
        # print("before board_positions[dst_y] = ",board_positions[dst_y])

        lines[dst_y][dst_x] = lines[src_y][src_x]  # 将src的子走到dst位置
        lines[src_y][src_x] = '1'    # 走子后，src位置赋值为空白

        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])

        # src_str = list(board_positions[src_y])
        # dst_str = list(board_positions[dst_y])
        # print("src_str[src_x] = ", src_str[src_x])
        # print("dst_str[dst_x] = ", dst_str[dst_x])
        # c = copy.deepcopy(src_str[src_x])
        # dst_str[dst_x] = c
        # src_str[src_x] = '1'
        # board_positions[dst_y] = ''.join(dst_str)
        # board_positions[src_y] = ''.join(src_str)
        # print("after board_positions[dst_y] = ", board_positions[dst_y])

        # board_positions[dst_y][dst_x] = board_positions[src_y][src_x]
        # board_positions[src_y][src_x] = '1'

        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")

        # GameBoard.print_borad(board)
        return board

    #将空格转为相当数量的数字1
    @staticmethod
    def board_to_pos_name(board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")

    @staticmethod
    def check_bounds(toY, toX):
        if toY < 0 or toX < 0:
            return False

        if toY >= BOARD_HEIGHT or toX >= BOARD_WIDTH:
            return False

        return True

    # upper为True自己为黑方，否则为红方
    @staticmethod
    def validate_move(c, upper=True):
        if (c.isalpha()):
            if (upper == True):
                if (c.islower()):
                    return True
                else:
                    return False
            else:
                if (c.isupper()):
                    return True
                else:
                    return False
        else:
            return True

    @staticmethod
    def get_legal_moves(state, current_player):
        moves = []
        k_x = None  # 黑方将军（将）走子
        k_y = None

        K_x = None  # 红方将军（帅）走子
        K_y = None

        face_to_face = False

        board_positions = np.array(GameBoard.board_to_pos_name(state))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if(board_positions[y][x].isalpha()):
                    if(board_positions[y][x] == 'r' and current_player == 'b'): # 黑方车走子
                        toY = y
                        for toX in range(x - 1, -1, -1):  # 往左走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):   # 车走到有棋子的地方
                                if (board_positions[toY][toX].isupper()):  # 有棋子的地方是红方，则表示吃红方的子
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, BOARD_WIDTH): # 往右走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1): # 往上走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, BOARD_HEIGHT): # 往下走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif(board_positions[y][x] == 'R' and current_player == 'w'):  # 红方车走子
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):   
                                if (board_positions[toY][toX].islower()):   # 红方车吃掉对方的子
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, BOARD_WIDTH):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, BOARD_HEIGHT):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif ((board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b'):  # 黑方马走子
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):  
                                toY = y + 2 * i   # 向纵方向走“日字”
                                toX = x + 1 * j
                                # GameBoard.validate_move(board_positions[toY][toX], upper=False)判断目标点是否为对方棋子
                                # board_positions[toY - i][x].isalpha() == False 判断马将要落子方向上是否有棋子挡住不能走
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                                toY = y + 1 * i   # 向横方向走“日字”
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif ((board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'w'):  # 红方马走子
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif ((board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b'):  # 黑方象走子
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i
                            # toY >= 5，限定黑方象只能在5~9行走（黑方在棋盘下方，从0行开始计算）
                            # board_positions[y + i // 2][x + i // 2].isalpha() == False，象前进方向上是否有子挡住不能走
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                            
                            toY = y + i
                            toX = x - i
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif ((board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'w'):  # 红方象走子
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif (board_positions[y][x] == 'a' and current_player == 'b'):  # 黑方士走子
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i
                            # toY >= 7 and toX >= 3 and toX <= 5， 限定黑方士在棋盘可移动坐标范围
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif (board_positions[y][x] == 'A' and current_player == 'w'):  # 红方士走子
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif (board_positions[y][x] == 'k'): 
                        k_x = x
                        k_y = y

                        if(current_player == 'b'):  # 黑方将走子
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign  # i为0表示横向移
                                    toX = x + j * sign  # j为0表示纵向移

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif (board_positions[y][x] == 'K'):
                        K_x = x
                        K_y = y

                        if(current_player == 'w'):  # 红方帅走子
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])
                    elif (board_positions[y][x] == 'c' and current_player == 'b'):  # 黑方炮走子
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):  # 往左方向走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()): # 若目标点有子
                                    hits = True
                                else:
                                    moves.append(m)  
                            else:   # 循环的上一次扫描时发现有子挡在前进方向上
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):  # 若目标点是对方的子，可以吃掉
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, BOARD_WIDTH): # 往右方向走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):  # 往上方向走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):  # 往下方向走
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'C' and current_player == 'w'):  # 红方炮走子
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, BOARD_WIDTH):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, BOARD_HEIGHT):
                            m = GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'p' and current_player == 'b'):  # 黑方卒走子
                        toY = y - 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                            moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                        if y < 5: # 已经过河
                            toY = y
                            toX = x + 1 # 可以往右横走
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                            toX = x - 1 # 可以往左横走
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                    elif (board_positions[y][x] == 'P' and current_player == 'w'):  # 红方兵走子
                        toY = y + 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                            moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

                            toX = x - 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.BOARD_POS_NAME[y][x] + GameBoard.BOARD_POS_NAME[toY][toX])

        if(K_x != None and k_x != None and K_x == k_x): # 当黑方将和红方帅面对面
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):  # K_y 是红方的，序号肯定比k_y小
                if(board_positions[i][K_x].isalpha()):  # 判断黑方将和红方帅之间有没其他子阻挡
                    face_to_face = False

        if(face_to_face == True):
            if(current_player == 'b'):  # 直杀对方将（帅）
                moves.append(GameBoard.BOARD_POS_NAME[k_y][k_x] + GameBoard.BOARD_POS_NAME[K_y][K_x])
            else:
                moves.append(GameBoard.BOARD_POS_NAME[K_y][K_x] + GameBoard.BOARD_POS_NAME[k_y][k_x])

        return moves