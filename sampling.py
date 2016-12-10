"""
Sampling states from a future state distribution
"""

import numpy as np


class SamplingHandler():

    def __init__(self, game, dist, policy, gamma):
        self.game = game
        self.dist = dist
        self.policy = policy
        self.gamma = gamma

    def acceptState(self):
        """
        Return true if the state is accepted
        With prob. self.gamma return true
        """
        return np.random.random() > self.gamma

    def getStartState(self):
        """
        Return a state
        Get a start state from self.dist
        """
        return self.game.getRestartState()
        # return np.random.choice(self.dist.key(), 1, self.dist.value())

    def getSampledState(self, startState, cutOff):
        """
        Return a sampled state
        Get a sample state from a rollout with prob. self.gamma
        """
        iteration = 1
        currState = startState
        while iteration <= cutOff:
            action = self.policy.getAction(currState)
            currState = self.game.getNextStateWithAction(currState, self.policy.getAction(currState))
            if currState.isTerminal():
                currState = startState
            elif self.acceptState():
                return currState
            iteration = iteration + 1

        return currState

    def sampleStates(self, stateNum, cutOff):
        """
        Return list of states
        Sample stateNum states.
        In each rollout, at most cutOff steps will be excuted.
        """
        sampledStates = []
        for i in range(stateNum):
            startState = self.getStartState()
            sampledState = self.getSampledState(startState, cutOff)
            sampledStates.append(sampledState)
            # print(sampledState)

        return sampledStates
