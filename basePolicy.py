from collections import Counter


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
            stateFeature = [self.game.state2feature(state)]
            return max(possibleActionList, key=lambda x: self.greedyPolicy[x].predict(stateFeature))
        else:
            return None

    def getActionsWithProb(self, state):
        bestAction = self.getAction(state)
        actionProbDict = Counter()
        actionProbDict[bestAction] = 1.0
        return actionProbDict
