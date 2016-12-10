
class Env(object):
    def __init__(self):
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

    def state2feature(self, state):
        pass

    def action2feature(self, action):
        pass
