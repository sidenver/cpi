

class game(object):

    def __init__(self, name=None):
        pass

    def getNextStateWithAction(self, state, action):
        pass

    def getPossibleActions(self, state):
        pass

    def getRestartState(self):
        pass

    def getReward(self, state, action, nextState):
        pass

    def isTerminal(self, state):
        pass
