from AlphaBetaPlayer import AlphaBetaPlayer
from time import time as time_fn

class LiteAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        super().__init__()


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
            # weight = 2
            heuristic_start_time = time_fn()
            loc_in_board = self.get_loc_in_board(board, is_self_player)
            simple_score = self.simple_state_score(board, loc_in_board)
            enemy_loc_in_board = self.get_loc_in_board(board, not is_self_player)
            simple_rival_score = self.simple_state_score(board, enemy_loc_in_board)
            simple_rival_score_improved = 10 if simple_rival_score == -1 else simple_rival_score
            heuristics = simple_score + simple_rival_score_improved
            curr_heuristic_time = time_fn() - heuristic_start_time
            self.heuristic_time = max(curr_heuristic_time, self.heuristic_time)
        return heuristics if is_self_player else -heuristics