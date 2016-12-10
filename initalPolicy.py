from collections import Counter
import numpy as np


class InitalPolicy(object):
    def __init__(self, env):
        self.env = env

    def getAction(self, state):
        possibleActionList = self.env.getPossibleActions(state)
        if not len(possibleActionList) == 0:
            return np.random.choice(possibleActionList, 1)[0]
        else:
            return None

    def getActionsWithProb(self, state):
        possibleActionList = self.env.getPossibleActions(state)
        actionProbDict = Counter()
        if not len(possibleActionList) == 0:
            for action in possibleActionList:
                actionProbDict[action] = 1.0/len(possibleActionList)
        return actionProbDict
