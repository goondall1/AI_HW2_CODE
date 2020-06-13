from AlphaBetaPlayer import AlphaBetaPlayer
from time import time as time_fn

class OrderedAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        super().__init__()
        self.last_iter_moves_values = {}

    def make_move(self, time=float('inf')):  # time parameter is not used, we assume we have enough time.
        # time_predictions = []
        ID_start_time = time_fn()
        self.last_iter_moves_values = {}
        self.iter_num = 1

        assert self.count_ones(self.board) == 1
        prev_loc = self.loc
        is_self_player = True
        depth = 1

        best_move, minimax_value, best_new_loc = None, float('-inf'), None
        best_move, minimax_value = self.alpha_beta(self.board.copy(), is_self_player, depth)
        last_iteration_time = time_fn() - ID_start_time
        next_iter_time_limit = self.get_iter_time_prediction(last_iteration_time)
        time_until_now = time_fn() - ID_start_time

        while time_until_now + next_iter_time_limit < time and depth < self.board.shape[0] * self.board.shape[1]:
            iteration_start_time = time_fn()
            depth += 1
            self.iter_num += 1
            self.prev_iter_leaves_developed = self.curr_iter_leaves_developed
            self.curr_iter_leaves_developed = 0
            best_move, minimax_value = self.alpha_beta(self.board.copy(), is_self_player, depth)

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

    def alpha_beta(self, board: list, is_self_player: bool, depth, alpha=float('-inf'), beta=float('inf')):
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

        is_root = self.iter_num == depth
        if is_root and self.last_iter_moves_values:
            ordered_moves = sorted(self.last_iter_moves_values, key=self.last_iter_moves_values.get, reverse=True)
            indices = [moves.index(move) for move in ordered_moves if move in moves]
            successors = [successors[i] for i in indices]
            moves = [moves[i] for i in indices]

        if is_self_player:
            current_max = float('-inf')
            current_move = None
            moves_values_dict = {}
            for child_board, move in zip(successors, moves):
                minimax_move, minimax_value = self.alpha_beta(child_board, not is_self_player, depth - 1, alpha, beta)
                if is_root:
                    moves_values_dict[move] = minimax_value
                if minimax_value > current_max:
                    current_max = minimax_value
                    current_move = move
                    alpha = max(current_max, alpha)
                    if current_max >= beta:
                        return None, float('inf')

            self.last_iter_moves_values = moves_values_dict
            return current_move, current_max
        else:
            current_min = float('inf')
            current_move = None
            for child_board, move in zip(successors, moves):
                minimax_move, minimax_value = self.alpha_beta(child_board, not is_self_player, depth - 1, alpha, beta)
                if minimax_value < current_min:
                    current_min = minimax_value
                    current_move = move
                    beta = min(current_min, beta)
                    if current_min <= alpha:
                        return None, float('-inf')
            return current_move, current_min
