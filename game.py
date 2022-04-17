import random
import math
from State import State
from Random import random_action

if __name__ == '__main__':
    # 状態の生成
    state = State()

    # ゲーム終了までのループ
    while True:
        if state.is_done():
            break

        # 次の状態の取得
        state = state.next(random_action(state))

        print(state)
        print()
