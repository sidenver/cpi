
class BasePolicy(object):
    def __init__(self, game, policyType=None):
        self.game = game
        if policyType is None:
            self.policy = {}

    def getAction(self, state):
        return self.policy[state]

    def getActionsWithProb(self, state):
        actionList = self.game.getPossibleActions(state)
        return {action: 1.0 if self.getAction(state) == action else 0.0 for action in actionList}
