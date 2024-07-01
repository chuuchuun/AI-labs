import math
from copy import deepcopy
import copy, sys

from lab3.exceptions import AgentException


def basic_static_eval(connect4, player="o"):
    four_player_counter = 0
    four_enemy_counter = 0
    if player == "o":
        enemy = "x"
    else:
        enemy = "o"
    for four in connect4.iter_fours():
        num_player_counter = four.count(player)
        num_enemy_counter = four.count(enemy)
        if num_player_counter == 3:
            four_player_counter += 1
        if num_enemy_counter == 3:
            four_enemy_counter += 1
    if four_enemy_counter+four_player_counter == 0:
        return 0
    return (four_player_counter - four_enemy_counter)/(four_enemy_counter+four_player_counter)


def advanced_static_eval(connect4, player="o"):
    scores = [
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 7, 11, 13, 11, 7, 5],
        [5, 7, 11, 13, 11, 7, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3]
    ]
    ai_score = 0
    human_score = 0
    for y in range(connect4.height):
        for x in range(connect4.width):
            if connect4.board[y][x] == player:
                ai_score += scores[y][x]
            elif connect4.board[y][x] is not None:
                human_score += scores[y][x]
    return (ai_score - human_score) / (ai_score + human_score)


class AlphaBetaAgent:
    def __init__(self, my_token="o", heuristic_func=basic_static_eval):
        self.my_token = my_token
        self.heuristic_func = heuristic_func

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")
        alpha = -math.inf
        beta = math.inf
        best_move, best_score = self.alphabeta(connect4, alpha, beta)
        return best_move

    def alphabeta(self, connect4, alpha, beta, depth=4, maximizing=True):
        best_move = None
        best_score = 0
        if maximizing is True:
            best_score = -math.inf
        else:
            best_score = math.inf
        new_board = deepcopy(connect4)
        for move in new_board.possible_drops():
            new_board = deepcopy(connect4)
            new_board.drop_token(move)
            v = 0
            if new_board.wins is not None:
                if new_board.wins == self.my_token:
                    v = 1
                else:
                    v = -1
            elif new_board.wins is None and new_board.game_over is True:
                v = 0
            elif depth == 0:
                v = self.heuristic_func(new_board, self.my_token)
            elif maximizing is True:
                v = self.alphabeta(new_board, alpha, beta, depth - 1, False)[1]
                alpha = max(alpha, v)
                if v >= beta:
                    break
            elif maximizing is False:
                if self.my_token == 'o':
                    new_board.who_moves = 'x'
                else:
                    new_board.who_moves = 'o'
                v = self.alphabeta(new_board,  alpha, beta, depth - 1, True)[1]
                beta = min(beta, v)
                if v <= alpha:
                    break
            if v > best_score and maximizing is True:
                best_score = v
                best_move = move
            if v < best_score and maximizing is False:
                best_score = v
                best_move = move
        return best_move, best_score
