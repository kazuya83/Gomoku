from State import State
from PvMcts import pv_mcts_scores
from DualNetwork import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os

# パラメータの準備
SP_GAME_COUNT = 500  # セルフプレイを行うゲーム数 ※普通は2500
SP_TEMPEARTURE = 1.0  # ボルツマン分布の温度パラメータ


def first_player_value(ended_state):
    # 1:先手勝利、-1:先手敗北、0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def play(model):
    # 学習データ
    history = []

    # 状態の生成
    state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores = pv_mcts_scores(model, state, SP_TEMPEARTURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces, policies, None]])

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def self_player():
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    model = load_model('./model/best.h5')

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model, i)
        history.append(h)

        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    # 学習データの保存
    write_data(history)

    # モデルの破棄
    K.clear_session()
    del model


if __name__ == '__main__':
    self_player()
