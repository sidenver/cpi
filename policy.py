from collections import Counter
from sklearn import svm
from basePolicy import BasePolicy


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

    def initalizedPolicy(self):
        X = []
        Y = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
        for x in range(10):
            state = self.game.getRestartState()
            action = self.game.getPossibleActions(state)[0]
            stateFeature = self.game.state2feature(state)
            actionFeature = self.game.action2feature(action)
            X.append(stateFeature+actionFeature)
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X, Y)
        initialPolicy = BasePolicy(lin_clf, self.game)
        self.policy[initialPolicy] = 1

    def conservativeUpdate(self, newPolicy, alpha=1.0):
        self.policy[newPolicy] = alpha / (1 - alpha)
        self.normalizePolicyWeight()
