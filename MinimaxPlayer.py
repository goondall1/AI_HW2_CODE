from time import time as time_fn


class MinimaxPlayer:
    def __init__(self):
        self.rival_loc = None
        self.loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.iter_num = 0
        self.dfs_depth = 10
        self.dfs_score = 0
        self.heuristic_time = 0
        self.curr_iter_leaves_developed = 0
        self.prev_iter_leaves_developed = 0

    def set_game_params(self, board):
        self.board = board
        for i, row in enumerate(board):  # i is row num
            for j, val in enumerate(row):  # j is row num
                if val == 1:
                    self.loc = (i, j)
                    # break
                elif val == 2:
                    self.rival_loc = (i, j)

    def simple_state_score(self, board, loc):
        num_steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available  # the score is higher if there is less steps available

    @staticmethod
    def count_ones(board):
        counter = 0
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    counter += 1
        return counter

    def make_move(self, time=float('inf')):  # time parameter is not used, we assume we have enough time.
        ID_start_time = time_fn()
        self.iter_num = 1

        assert self.count_ones(self.board) == 1
        prev_loc = self.loc
        is_self_player = True
        depth = 1

        best_move, minimax_value, best_new_loc = None, float('-inf'), None
        best_move, minimax_value = self.RB_minimax(self.board.copy(), is_self_player, depth)
        last_iteration_time = time_fn() - ID_start_time
        next_iter_time_limit = self.get_iter_time_prediction(last_iteration_time)
        time_until_now = time_fn() - ID_start_time

        while time_until_now + next_iter_time_limit < time and depth < self.board.shape[0] * self.board.shape[1]:
            iteration_start_time = time_fn()
            depth += 1
            self.prev_iter_leaves_developed = self.curr_iter_leaves_developed
            self.curr_iter_leaves_developed = 0
            self.iter_num += 1
            best_move, minimax_value = self.RB_minimax(self.board.copy(), is_self_player, depth)


            if minimax_value == 1000 or minimax_value == -1000:
                break

            last_iteration_time = time_fn() - iteration_start_time
            next_iter_time_limit = self.get_iter_time_prediction(last_iteration_time)
            time_until_now = time_fn() - ID_start_time

        if best_move is None:
            exit()

        best_new_loc = (prev_loc[0] + best_move[0], prev_loc[1] + best_move[1])
        self.board[best_new_loc] = 1
        self.board[prev_loc] = -1

        assert self.count_ones(self.board) == 1

        self.loc = best_new_loc
        print('returning move', best_move)
        time_until_now = time_fn() - ID_start_time
        return best_move
        # return depth

    def get_iter_time_prediction(self, last_iteration_time):
        if self.iter_num == 1:
            return 4 * last_iteration_time
        else:
            heuristic_time_limit = self.heuristic_time
            last_iter_added_leaves = self.curr_iter_leaves_developed - self.prev_iter_leaves_developed
            return 3 * last_iteration_time + last_iter_added_leaves * heuristic_time_limit

    def RB_minimax(self, board: list, is_self_player: bool, depth):
        if self.is_board_final(board, is_self_player) or depth == 0:
            self.curr_iter_leaves_developed += 1
            leaf_score = self.heuristic(board, is_self_player)
            move = None
            if leaf_score == 1000:
                boards, moves = self.get_board_successors(self.board, is_self_player)
                if moves:
                    move = moves[0]  # one of any in case of win
            return move, leaf_score

        successors, moves = self.get_board_successors(board, is_self_player)

        assert len(successors), "past final check but still no legal steps: "

        if is_self_player:
            current_max = float('-inf')
            current_move = None
            for child_board, move in zip(successors, moves):
                minimax_move, minimax_value = self.RB_minimax(child_board, not is_self_player, depth - 1)
                if minimax_value > current_max:
                    current_max = minimax_value
                    current_move = move
            return current_move, current_max

        else:
            current_min = float('inf')
            current_move = None
            for child_board, move in zip(successors, moves):
                minimax_move, minimax_value = self.RB_minimax(child_board, not is_self_player, depth - 1)
                if minimax_value < current_min:
                    current_min = minimax_value
                    current_move = move
            return current_move, current_min

    def is_board_final(self, board: list, is_self_player: bool) -> bool:
        player1_legal_successors_num, moves1 = self.get_board_successors(board, is_self_player)
        return len(player1_legal_successors_num) == 0

    def get_board_successors(self, board: list, is_self_player: bool):
        successors, moves = [], []
        prev_loc = self.get_loc_in_board(board, is_self_player)

        for d in self.directions:
            i = prev_loc[0] + d[0]
            j = prev_loc[1] + d[1]

            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                new_board = board.copy()
                new_board[prev_loc] = -1
                new_board[i, j] = 1 if is_self_player else 2
                successors.append(new_board)
                moves.append(d)
        return successors, moves

    def heuristic(self, board: list, is_self_player):
        # if the board presents ending state for self_player
        # Utility function:
        # (is_final, self_player_succ, rival_player_succ) = self.is_board_final(board, is_sel_player)
        is_final = self.is_board_final(board, is_self_player)
        rival_player_succ, rival_moves = self.get_board_successors(board, not is_self_player)
        if is_final and is_self_player:
            # is_self_player == True -> meaning, final board is due to self_player have no moves
            # in this case only tie or a lose is posiible.
            if len(rival_player_succ) == 0:
                return 0
            else:
                return -1000
        elif is_final and not is_self_player:
            # is_self_player == False -> meaning, final board is due to rival_player have no moves
            # in this case only tie or a win is posiible.
            # a win is possible if self_player successors have moves ( i.e they have legal successors.
            second_level_succ_num = 0
            self_player_succ = rival_player_succ
            if len(self_player_succ) > 0:
                second_level_succ_num = sum(
                    [len(self.get_board_successors(succ_board, not is_self_player)[0]) for succ_board in
                     self_player_succ])
                if second_level_succ_num > 0:
                    return 1000
            return 0
        # Heuristic Function:
        else:
            heuristic_start_time = time_fn()
            loc_in_board = self.get_loc_in_board(board, is_self_player)
            simple_score = self.simple_state_score(board, loc_in_board)
            enemy_loc_in_board = self.get_loc_in_board(board, not is_self_player)
            simple_rival_score_improved = 0

            manhatan_dist = abs(loc_in_board[0] - enemy_loc_in_board[0]) + abs(loc_in_board[1] - enemy_loc_in_board[1])
            assert manhatan_dist != 0, "players in the same cell"
            manhatan_score = 1 / manhatan_dist

            dfs_score = 0
            if manhatan_score < self.dfs_depth:
                simple_rival_score = self.simple_state_score(board, enemy_loc_in_board)
                simple_rival_score_improved = 10 if simple_rival_score == -1 else simple_rival_score
                dfs_cells = self.dfs(board, loc_in_board, self.dfs_depth)
                dfs_rival_cells = self.dfs(board, enemy_loc_in_board, self.dfs_depth)
                dfs_score = len(dfs_cells) - len(dfs_rival_cells)

            heuristics = simple_score + simple_rival_score_improved + dfs_score + manhatan_score
            curr_heuristic_time = time_fn() - heuristic_start_time
            self.heuristic_time = max(curr_heuristic_time, self.heuristic_time)
        return heuristics if is_self_player else -heuristics

    @staticmethod
    def get_loc_in_board(board: list, is_self_player: bool):
        player = 1 if is_self_player else 2
        for i, row in enumerate(board):  # i is row num
            for j, val in enumerate(row):  # j is col num
                if player == val:
                    return i, j

    def set_rival_move(self, loc):
        self.board[self.rival_loc] = -1
        self.board[loc] = 2
        self.rival_loc = loc

    def dfs(self, board, loc, dfs_depth):
        dfs_cells = set()
        is_self_player = board[loc] == 1
        if dfs_depth == 0:
            return dfs_cells
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]

            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:
                dfs_cells.add((i, j))
                new_board = board.copy()
                new_board[loc] = -1
                new_board[i, j] = 1 if is_self_player else 2
                dfs_cells |= self.dfs(new_board, (i, j), dfs_depth - 1)
        return dfs_cells
