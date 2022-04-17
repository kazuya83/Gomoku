import math
import CommonProc

N = 100


def mcts_action(state):
    # モンテカルロ木探索のノードの定義
    class Node:
        def __init__(self, state):
            self.state = state
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 局面の価値を計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = CommonProc.playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                    return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で勝ちを取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w = value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        # UCB1が最大の子ノードの取得
        def next_child_node(self):
            # 試行回数が0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n +
                                   (2*math.log(t)/child_node.n)**0.5)
            # UCB1が最大の子ノードを返す
            return self.child_nodes[CommonProc.argmax(ucb1_values)]

    # 現在の局面のノードの作成
    root_node = Node(state)
    root_node.expand()

    # N回のシミュレーションを実行
    for _ in range(N):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[CommonProc.argmax(n_list)]
