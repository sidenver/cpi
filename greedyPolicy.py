from basePolicy import BasePolicy
from sklearn import svm
from collections import defaultdict


class GreedyPolicy(object):
    def __init__(self, game):
        self.game = game

    def getGreedyPolicy(self, stateActionQList):
        stateToActionQList = defaultdict(list)
        for state, action, estimateQ in stateActionQList:
            stateToActionQList[state].append((action, estimateQ))
        X = []
        Y = []
        for state in stateToActionQList:
            maxActionQ = max(stateToActionQList[state], key=lambda x: x[1])
            for action, estimateQ in stateToActionQList[state]:
                stateFeature = self.game.state2feature(state)
                actionFeature = self.game.action2feature(action)
                feature = stateFeature + actionFeature
                X.append(feature)
                if action == maxActionQ[0]:
                    Y.append(1)
                else:
                    Y.append(0)
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X, Y)
        greedyPolicy = BasePolicy(lin_clf, self.game)
        return greedyPolicy
