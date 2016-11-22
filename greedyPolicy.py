from basePolicy import BasePolicy


class GreedyPolicy(object):
    def __init__(self, game):
        self.game = game

    def getGreedyPolicy(self, stateActionQList):
        greedyPolicy = BasePolicy(self.game)
        # do some ML and update
        return greedyPolicy
