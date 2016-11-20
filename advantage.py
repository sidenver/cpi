"""
Estimate advantage.
First, sampling a set of states and its action, 
Second, estimate q-value of the state-action pair.
Third, use estimated q-value to get a new policy.
Last, calculate the advatange between new policy and old policy
and return the new policy and the advantage.
"""

import numpy as np
import sampling


class AdvantageEstimator():

    def __init__(self, game, dist, discount, gamma):
        self.game = game
        self.dist = dist
        self.discount = discount
        self.gamma = gamma

    def getRandomAction(self, actions):
    	"""
        Return an action
        Uniformly get a random action
        """
        return np.random.random(actions, 1)

    def estimatQValue(self, policy, state, action, horizon = 30):
        """
        Return a q-value
        Estimate q-value from a single roll out and reward of next state (state + action)
        """
        nextState = self.game.getNextStateWithAction(state, action)
        reward = self.game.getReward(state, action, nextState)
        state = nextState
        for i in range(horizon - 1):
            if (self.game.isTerminal(state)):
                break
            action = policy.getAction(state)
            nextState = self.game.getNextStateWithAction(state, action)
            reward += self.discount * self.game.getReward(state, action, nextState)
            state = nextState

        return reward * (1 - self.discount)

    def getSampledStateActionQ(self, policy, stateNum, cutOff):
        """
        Return a {s, a, q} set with size #stateNum
        Sample #stateNum triplets (state, action, q-value)
        """
        stateActionQList = []
        samplingHandler = sampling.SamplingHandler(self.game, self.dist, policy, self.gamma)
        sampledStates = samplingHandler.sampledStates(stateNum, cutOff)
        for state in sampledStates:
            actions = self.game.getPossibleActions(state)
            action = self.getRandomAction(actions)
            qValue = self.estimateQValue(policy, state, action)
            stateActionQList.append((state, action, qValue))

        return stateActionQList

    def estimateAdvantage(self, policy, stateNum, cutOff):
        """
        Return a new policy and it's advantage
        Sample #stateNum states, get an action randomly for each state
        Calculate sampled q-value, then get a new policy
        Last, estimate the advantage
        """
        sampledSet = self.getSampledStateActionQ(policy, stateNum, cutOff)
        newPolicy = []  # getNewPolicy(sampledSet)
        advantage = 0.0
        for sample in sampledSet:
            numOfActions = len(self.game.getPossibleActions(sample[0]))
            actionProbDiff = newPolicy.getProb(sample[0], sample[1]) - policy.getProb(sample[0], sample[1])
            advantage += numOfActions * sample[2] * actionProbDiff

        return {'newPolicy': newPolicy, 'advantage': (advantage / stateNum)}
