from collections import Counter


class Policy(object):
    def __init__(self, game):
        self.game = game
        self.policy = Counter()

    def getAction(self, state):
        actionProbDict = self.getActionsWithProb(state)
        actionList = self.game.getPossibleActions(state)
        return max(actionList, key=lambda x: actionProbDict[x])

    def getActionsWithProb(self, state):
        # {action: prob}
        actionProbDict = Counter()
        for basePolicy in self.policy:
            predictedAction = basePolicy.getAction(state)
            actionProbDict[predictedAction] += self.policy[basePolicy]

        return actionProbDict

    def normalizePolicyWeight(self):
        total = sum(self.policy.values(), 0.0)
        for key in self.policy:
            self.policy[key] /= total

    def conservativeUpdate(self, newPolicy, alpha=1.0):
        if alpha == 1.0:
            self.policy[newPolicy] = 1.0
        else:
            self.policy[newPolicy] = alpha / (1 - alpha)
            self.normalizePolicyWeight()
