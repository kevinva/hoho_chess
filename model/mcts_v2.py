

class leaf_node(object):
    def __init__(self, in_parent, in_prior_p, in_state):
        self.P = in_prior_p
        self.Q = 0
        self.N = 0
        self.v = 0
        self.U = 0
        self.W = 0
        self.parent = in_parent
        self.child = {}
        self.state = in_state

    def is_leaf(self):
        return self.child == {}

    def get_Q_plus_U_new(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        U = c_puct * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        return self.Q + U

    def get_Q_plus_U(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        return self.Q + self.U

    # def select_move_by_action_score(self, noise=True):
    #
    #     # P = params[self.lookup['P']]
    #     # N = params[self.lookup['N']]
    #     # Q = params[self.lookup['W']] / (N + 1e-8)
    #     # U = c_PUCT * P * np.sqrt(np.sum(N)) / (1 + N)
    #
    #     ret_a = None
    #     ret_n = None
    #     action_idx = {}
    #     action_score = []
    #     i = 0
    #     for a, n in self.child.items():
    #         U = c_PUCT * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
    #         action_idx[i] = (a, n)
    #
    #         if noise:
    #             action_score.append(n.Q + U * (0.75 * n.P + 0.25 * dirichlet([.03] * (go.N ** 2 + 1))) / (n.P + 1e-8))
    #         else:
    #             action_score.append(n.Q + U)
    #         i += 1
    #         # if(n.Q + n.U > max_Q_plus_U):
    #         #     max_Q_plus_U = n.Q + n.U
    #         #     ret_a = a
    #         #     ret_n = n
    #
    #     action_t = int(np.argmax(action_score[:-1]))
    #
    #     return ret_a, ret_n
    #     # return action_t
    def select_new(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U_new(c_puct))

    def select(self, c_puct):
        # max_Q_plus_U = 1e-10
        # ret_a = None
        # ret_n = None
        # for a, n in self.child.items():
        #     n.U = c_puct * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
        #     if(n.Q + n.U > max_Q_plus_U):
        #         max_Q_plus_U = n.Q + n.U
        #         ret_a = a
        #         ret_n = n
        # return ret_a, ret_n
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))

    #@profile
    # moves为所有合法走子，
    def expand(self, moves, action_probs):
        tot_p = 1e-8
        action_probs = action_probs.flatten()   #.squeeze()
        # print("expand action_probs shape : ", action_probs.shape)
        for action in moves:
            in_state = GameBoard.sim_do_action(action, self.state)  # 字符串棋盘
            mov_p = action_probs[label2i[action]]
            new_node = leaf_node(self, mov_p, in_state)
            self.child[action] = new_node
            tot_p += mov_p

        for a, n in self.child.items():
            n.P /= tot_p    # hoho: 为啥还要除一下？

    # 只更新自己
    def back_up_value(self, value):
        self.N += 1
        self.W += value
        self.v = value
        self.Q = self.W / self.N  # node.Q += 1.0*(value - node.Q) / node.N
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        # node = node.parent
        # value = -value

    # 更新自己，一直回溯到根节点
    def backup(self, value):
        node = self
        while node != None:
            node.N += 1
            node.W += value
            node.v = value
            node.Q = node.W / node.N    # node.Q += 1.0*(value - node.Q) / node.N
            node = node.parent
            value = -value