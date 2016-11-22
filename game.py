import gym
env = gym.make('MountainCar-v0')
class game(object):

    def __init__(self, name=None):
        pass

    def getNextStateWithAction(self, state, action):
        action = getPossibleActions(self.state)
        self.state, reward, done, info = env._step(action)
        return self.state

    def getPossibleActions(self, state):
        self.action = env.action_space.sample()
        return self.action
        
    def getRestartState(self):
        pass

    def getReward(self, state, action, nextState):
        pass

    def isTerminal(self, state):
        pass
