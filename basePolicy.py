from collections import Counter
import numpy as np


class BasePolicy(object):
    def __init__(self, greedyPolicy, game):
        self.greedyPolicy = greedyPolicy
        self.game = game

    def getAction(self, state):
        possibleActionList = self.game.getPossibleActions(state)
        if not len(possibleActionList) == 0:
            # stateFeature = self.game.state2feature(state)
            # X = [stateFeature + self.game.action2feature(action) for action in possibleActionList]
            # confidence_scores = self.greedyPolicy.decision_function(X)
            # actionScores = {action: confidence_scores[idx] for idx, action in enumerate(possibleActionList)}
            # return max(possibleActionList, key=lambda x: actionScores[x])
            # actionProb = self.getActionsWithProb(state)
            # r = np.random.random()
            # total = 0.0
            # for action in actionProb:
            #     total += actionProb[action]
            #     if r <= total:
            #         return action
            stateFeature = [self.game.state2feature(state)]
            return max(possibleActionList, key=lambda x: self.greedyPolicy[x].predict(stateFeature))
        else:
            return None

    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def getActionsWithProb(self, state):
        # possibleActionList = self.game.getPossibleActions(state)
        # if not len(possibleActionList) == 0:
        #     stateFeature = [self.game.state2feature(state)]
        #     qActionList = [self.greedyPolicy[x].predict(stateFeature)[0] for x in possibleActionList]
        #     # print qActionList
        #     probList = self.softmax(qActionList)
        #     # print probList
        #     actionProbDict = Counter({action: prob for action, prob in zip(possibleActionList, probList)})
        #     return actionProbDict
        # else:
        #     return None

        bestAction = self.getAction(state)
        actionProbDict = Counter()
        actionProbDict[bestAction] = 1.0
        return actionProbDict
