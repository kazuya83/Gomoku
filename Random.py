import random


def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]
