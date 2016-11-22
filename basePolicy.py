
class BasePolicy(object):
    def __init__(self, policyType=None):
        self.miniPolicy = {}

    def getAction(self, state):
        return self.miniPolicy[state]
