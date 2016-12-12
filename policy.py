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

    def times(self, actionProb, weight):
        newActionProb = Counter()
        for action in actionProb:
            newActionProb[action] = actionProb[action] * weight
        return newActionProb

    def getActionsWithProb(self, state):
        # {action: prob}
        actionList = self.game.getPossibleActions(state)
        if len(actionList) == 0:
            return None
        actionProbDict = Counter()
        for basePolicy in self.policy:
            actionProb = basePolicy.getActionsWithProb(state)
            weightActionProb = self.times(actionProb, self.policy[basePolicy])
            actionProbDict.update(weightActionProb)
        self.normalizeProb(actionProbDict)
        return actionProbDict

        # for basePolicy in self.policy:
        #     predictedAction = basePolicy.getAction(state)
        #     if predictedAction in actionList:
        #         actionProbDict[predictedAction] += self.policy[basePolicy]
        # self.normalizeProb(actionProbDict)
        # return actionProbDict

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
            most_common = self.policy.most_common(20)
            self.policy = Counter({policy: weight for policy, weight in most_common})
            total = sum(self.policy.values(), 0.0)
            print 'total after triming is:', total
            self.policy[newPolicy] = alpha / (1 - alpha) * total
            self.normalizePolicyWeight()
        else:
            print 'bad alpha:', alpha
