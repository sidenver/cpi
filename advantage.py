"""
"""

import random
import numpy as np
import sampling

class AdvantageEstimator():

    def __init__(self, game, dist, policy, gamma):
        self.game = game
        self.dist = dist
        self.policy = policy
        self.gamma = gamma
        self.Sampling = Sampling(game, dist, policy, gamma)

    def getRandomAction(self, actions):
    	"""
    	Uniformly get a random action
    	"""
    	return np.random.random(actions, 1)

    def estimatQvalue(self, stateNum, cutOff):
    	sampledStates = self.Sampling.sampledStates(stateNum, cutOff)
        for state in sampledStates:
        	actions = self.game.getLegalActions(state)
        	action = self.getRandomAction(actions)

    def estimateAdvantage(self, stateNum, cutOff):
    	estimatedQvalue = self.estimateQvalue(stateNum, cutOff)
        
            