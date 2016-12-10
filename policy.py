from collections import Counter
from initalPolicy import InitalPolicy
import numpy as np


class Policy(object):
    def __init__(self, game):
        self.game = game
        self.policy = Counter()
        self.initalizedPolicy()

    def getAction(self, state):
        actionList = self.game.getPossibleActions(state)
        if len(actionList) == 0:
            return None
        actionProbDict = self.getActionsWithProb(state)
        action = np.random.choice(actionList, 1, p=[actionProbDict[action] for action in actionList])[0]

        return action
        # return max(actionList, key=lambda x: actionProbDict[x])

    def getActionsWithProb(self, state):
        # {action: prob}
        actionList = self.game.getPossibleActions(state)
        if len(actionList) == 0:
            return None
        actionProbDict = Counter()
        for basePolicy in self.policy:
            predictedAction = basePolicy.getAction(state)
            if predictedAction in actionList:
                actionProbDict[predictedAction] += self.policy[basePolicy]
        self.normalizeProb(actionProbDict)
        return actionProbDict

    def normalizeProb(self, actionProbDict):
        total = sum(actionProbDict.values(), 0.0)
        for key in actionProbDict:
            actionProbDict[key] /= total

    def normalizePolicyWeight(self):
        total = sum(self.policy.values(), 0.0)
        for key in self.policy:
            self.policy[key] /= total

    def initalizedPolicy(self):
        initialPolicy = InitalPolicy(self.game)
        self.policy[initialPolicy] = 1

    def conservativeUpdate(self, newPolicy, alpha=1.0):
        if alpha > 0 and alpha < 1:
            self.policy[newPolicy] = alpha / (1 - alpha)
            self.normalizePolicyWeight()
        else:
            print 'bad alpha:', alpha
