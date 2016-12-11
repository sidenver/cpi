from basePolicy import BasePolicy
from sklearn import linear_model, svm
from collections import defaultdict


class GreedyPolicy(object):
    def __init__(self, game):
        self.game = game

    def getGreedyPolicy(self, stateActionAdvantageList, mode='regression'):
        if mode == 'svm':
            stateToActionQList = defaultdict(list)
            for state, action, estimateQ in stateActionAdvantageList:
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
        elif mode == 'regression':
            actionToStateAList = defaultdict(list)
            for state, action, estimateAd in stateActionAdvantageList:
                actionToStateAList[action].append((state, estimateAd))
            regresserDict = {}
            X = defaultdict(list)
            Y = defaultdict(list)
            for action in actionToStateAList:
                for state, estimateAd in actionToStateAList[action]:
                    stateFeature = self.game.state2feature(state)
                    feature = stateFeature
                    X[action].append(feature)
                    Y[action].append(estimateAd)
                # regresserDict[action] = linear_model.LinearRegression()
                regresserDict[action] = linear_model.Ridge(alpha=.5)
                # regresserDict[action] = svm.SVR(C=1.0, epsilon=0.2)
                regresserDict[action].fit(X[action], Y[action])
            greedyPolicy = BasePolicy(regresserDict, self.game)
        return greedyPolicy
