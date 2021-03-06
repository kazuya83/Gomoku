import random

MAZE_CNT = 3


class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        # 石の配置
        self.pieces = pieces if pieces != None else [0] * (MAZE_CNT**2)
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [
            0] * (MAZE_CNT**2)

    # 石の数取得
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    # 負け判定
    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for k in range(MAZE_CNT):
                if y < 0 or MAZE_CNT-1 < y or x < 0 or MAZE_CNT-1 < x or self.enemy_pieces[x+y*(MAZE_CNT)] == 0:
                    return False
                x, y = x+dx, y+dy
            return True

        # 負け判定
        if is_comp(0, 0, 1, 1) or is_comp(0, MAZE_CNT-1, 1, -1):
            return True
        for i in range(MAZE_CNT):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    # 引き分け判定
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == MAZE_CNT**2

    # ゲーム終了判定
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # 次の状態取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    # 合法手のリスト取得
    def legal_actions(self):
        actions = []
        for i in range(MAZE_CNT**2):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    # 先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(MAZE_CNT**2):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '_'
            if i % 3 == 2:
                str += '\n'

        return str
