from Random import random_action


def argmax(collection, key=None):
    return collection.index(max(collection))


def playout(state):
    # 負け状態価値:-1
    if state.is_lose():
        return -1

    # 引き分け状態価値:0
    if state.is_draw():
        return 0

    # 次の状態価値
    return -playout(state.next(random_action(state)))
